"""从原始遥感影像读取局部窗口，转换提示和轮廓坐标。"""

import time
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window

from core.preview import _stretch_uint8_or_float
from core.sam_roi import validate_prompts
from core.sam_postprocess import process_regions, validate_postprocess
from core.sam_boundary import gap_kernel_for_width
from core.sam_scribble import validate_scribbles, sample_scribbles
from affine import Affine
from rasterio.features import rasterize


def _window_for_points(coords, width, height, size, padding, maximum):
    """保留所有提示点及上下文，只限制局部窗口，不读取整幅大影像。"""
    low, high = coords.min(axis=0), coords.max(axis=0)
    spans = np.ceil(high - low + 1).astype(int) + 2 * padding
    window_w = min(width, max(size, int(spans[0])))
    window_h = min(height, max(size, int(spans[1])))
    if max(window_w, window_h) > maximum:
        raise ValueError("提示点覆盖范围过大，请按田块分别识别，或增大配置中的 max_window_size")
    center = (low + high) / 2
    left = int(np.clip(np.floor(center[0] - window_w / 2), 0, width - window_w))
    top = int(np.clip(np.floor(center[1] - window_h / 2), 0, height - window_h))
    return Window(left, top, window_w, window_h)


def _check_grid(src, georef):
    if ((src.width, src.height) != (georef["source_width"], georef["source_height"])
            or not np.allclose(src.transform[:6], georef["transform"], rtol=0, atol=1e-12)
            or (src.crs.to_string() if src.crs else None) != georef.get("crs")):
        raise ValueError("原始影像网格与预览不一致，请重新校验并生成预览")


def _read_window(sources, window, composite):
    """按原生像素读取，不使用降采样或有损压缩；无效像元用白色显示。"""
    if composite:
        bands = [src.read(1, window=window) for src in sources]
        valid = np.logical_and.reduce([src.read_masks(1, window=window) > 0 for src in sources])
    else:
        src = sources[0]
        indexes = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        bands = list(src.read(indexes, window=window))
        if ColorInterp.alpha in src.colorinterp:
            alpha_index = src.colorinterp.index(ColorInterp.alpha) + 1
            valid = src.read(alpha_index, window=window) > 0
            if src.nodata is not None:
                valid &= ~np.logical_and.reduce([band == src.nodata for band in bands])
        else:
            valid = src.dataset_mask(window=window) > 0
    valid &= np.logical_and.reduce([np.isfinite(band) for band in bands])
    channels = []
    for band in bands:
        if band.dtype == np.uint8:
            channel = band.copy()
        else:
            # 拉伸仅统计有效像元，避免无数据值改变局部对比度。
            values = np.where(valid, band, np.nan)
            channel = _stretch_uint8_or_float(values)
        channel[~valid] = 255
        channels.append(channel)
    return np.stack(channels, axis=-1), valid


def _touches_crop(regions, window, width, height):
    """只检查人工裁剪边缘，原始影像的真实边缘不要求继续扩窗。"""
    points = np.concatenate([np.asarray(region["hull"]) for region in regions])
    low, high = points.min(axis=0), points.max(axis=0)
    return ((window.col_off > 0 and low[0] <= 1)
            or (window.row_off > 0 and low[1] <= 1)
            or (window.col_off + window.width < width and high[0] >= window.width - 1)
            or (window.row_off + window.height < height and high[1] >= window.height - 1))


def predict_tiff_roi(service, preview, points, excluded, model_id, options=None, postprocess=None, scribbles=None):
    """返回预览坐标轮廓；模型输入始终来自原始 TIFF 的局部像素。"""
    started = time.perf_counter()
    settings = validate_postprocess(postprocess)
    options = options or {}
    size = int(options.get("window_size", 2048))
    maximum = int(options.get("max_window_size", 8192))
    padding = int(options.get("padding", 256))
    if not 32 <= size <= maximum <= 16384 or not 0 <= padding < maximum / 2:
        raise ValueError("局部分割窗口配置无效")
    g = preview["georef"]
    coords, labels = validate_prompts(points, g["preview_width"], g["preview_height"])
    scale = np.array([g["source_width"] / g["preview_width"],
                      g["source_height"] / g["preview_height"]])
    # 与既有提示点地理坐标一致，取预览像元中心；最外侧点限制在源影像内。
    source_coords = np.minimum((coords.astype(np.float64) + 0.5) * scale,
                               [g["source_width"] - 0.001, g["source_height"] - 0.001])
    strokes = validate_scribbles(scribbles if scribbles is not None else [],
                                g["preview_width"], g["preview_height"])
    line_coords, line_extent = sample_scribbles(strokes, scale, 128 - len(points))
    original_count = len(source_coords)
    if len(line_coords):
        line_coords = np.minimum(line_coords, [g["source_width"] - 0.001, g["source_height"] - 0.001])
        negative_pixels = {tuple(p.astype(int)) for p in source_coords[labels == 0]}
        if any(tuple(p.astype(int)) in negative_pixels for p in line_coords):
            raise ValueError("划线保留提示与排除点冲突，请调整线提示或排除点")
        source_coords = np.concatenate([source_coords, line_coords])
        labels = np.r_[labels, np.ones(len(line_coords), dtype=np.int32)]
    extent = np.concatenate([source_coords, line_extent])
    paths = preview.get("source_paths") or {}
    composite = preview.get("mode") == "composite"
    if not composite and preview.get("mode") != "result_preview":
        raise ValueError("缺少原始影像来源，请重新校验影像")
    roles = ("B4", "B3", "B2") if composite else ("preview",)
    if any(role not in paths for role in roles):
        raise ValueError("缺少原始 TIFF 路径，请重新校验影像并生成预览")

    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(paths[role])) for role in roles]
        for src in sources:
            _check_grid(src, g)
        width, height = sources[0].width, sources[0].height
        attempts = 0
        while True:
            window = _window_for_points(extent, width, height, size, padding, maximum)
            origin = np.array([window.col_off, window.row_off])
            local_points = [{"x": float(x), "y": float(y), "label": int(label)}
                            for (x, y), label in zip(source_coords - origin, labels)]

            def local_ring(ring):
                return (np.asarray(ring, dtype=float) * scale - origin).tolist()

            local_excluded = [{"hull": local_ring(region["hull"]),
                               "holes": [local_ring(h) for h in region.get("holes", [])]}
                              for region in excluded]
            rgb, valid = _read_window(sources, window, composite)
            ix, iy = (source_coords - origin).astype(int).T
            if not valid[iy[labels == 1], ix[labels == 1]].all():
                raise ValueError("保留点落在原始影像无效像元上，请调整提示位置")
            if len(line_coords) and local_excluded:
                geometries = [({"type": "Polygon", "coordinates": [r["hull"], *r["holes"]]}, 1)
                              for r in local_excluded]
                blocked = rasterize(geometries, out_shape=valid.shape, transform=Affine.identity())
                if blocked[iy[original_count:], ix[original_count:]].any():
                    raise ValueError("划线提示落在已确认区域内，请调整划线位置")
            image_key = (preview["preview_id"], *window.flatten())
            gap_options = {}
            if settings["ignore_small_boundaries"]:
                gap_options["gap_kernel"] = gap_kernel_for_width(settings["max_gap_width_m"],
                                                                sources[0].window_transform(window),
                                                                sources[0].crs, valid.shape)
            result = service.predict(rgb, image_key, local_points, local_excluded,
                                     model_id=model_id, valid_mask=valid, **gap_options)
            attempts += 1
            if not _touches_crop(result["regions"], window, width, height):
                processed = process_regions(result["regions"], sources[0].window_transform(window),
                                            sources[0].crs, source_coords - origin, labels,
                                            valid, local_excluded, settings)
                if not _touches_crop(processed, window, width, height):
                    result["regions"] = processed
                    break
            next_size = min(maximum, max(int(window.width), int(window.height)) * 2)
            if next_size <= size or max(window.width, window.height) >= maximum:
                raise ValueError("识别区域仍触及局部窗口边缘，请添加排除点限制田块，或增大 max_window_size 后重试")
            size = next_size

    result["postprocess"] = settings
    result["scribble_count"] = len(strokes)
    result["scribble_prompt_count"] = len(line_coords)

    def preview_ring(ring):
        return ((np.asarray(ring, dtype=float) + origin) / scale).tolist()

    for region in result["regions"]:
        region["hull"] = preview_ring(region["hull"])
        region["holes"] = [preview_ring(hole) for hole in region["holes"]]
    result["inference_source"] = "original_tiff_window"
    result["inference_window"] = {"x": int(window.col_off), "y": int(window.row_off),
                                  "width": int(window.width), "height": int(window.height),
                                  "attempts": attempts}
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result
