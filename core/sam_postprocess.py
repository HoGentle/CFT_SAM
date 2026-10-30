"""半自动区域的去洞、小凹槽填补与外边界平滑。"""

import numpy as np
from affine import Affine
from PIL import Image, ImageFilter
from rasterio.features import rasterize, shapes
from rasterio.warp import transform as project

from core.prescription import utm_crs_for
from core.sam_roi import contains_point, mask_to_regions
from core.sam_boundary import close_mask, gap_kernel_for_width, _binary_filter_axis


def validate_postprocess(settings):
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("区域处理选项必须是对象")
    fill = settings.get("fill_holes", False)
    smooth = settings.get("smooth_boundary", False)
    ignore = settings.get("ignore_small_boundaries", False)
    notches = settings.get("fill_small_notches", False)
    if not all(isinstance(value, bool) for value in (fill, smooth, ignore, notches)):
        raise ValueError("去洞、平滑边界、忽略细小分隔和填补小凹槽选项必须是布尔值")
    area = settings.get("max_hole_area_m2", 10.0)
    if isinstance(area, bool) or not isinstance(area, (int, float)) or not np.isfinite(area) or area <= 0:
        raise ValueError("去洞最大面积必须是大于 0 的有限数值，单位为平方米")
    width = settings.get("max_gap_width_m", 0.2)
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not np.isfinite(width) or not 0 < width <= 5:
        raise ValueError("细小分隔宽度阈值必须大于 0 且不超过 5 米")
    notch_width = settings.get("max_notch_width_m", 0.5)
    if isinstance(notch_width, bool) or not isinstance(notch_width, (int, float)) or not np.isfinite(notch_width) or not 0 < notch_width <= 5:
        raise ValueError("凹槽槽口宽度必须大于 0 且不超过 5 米")
    notch_area = settings.get("max_notch_area_m2", 10.0)
    if isinstance(notch_area, bool) or not isinstance(notch_area, (int, float)) or not np.isfinite(notch_area) or notch_area <= 0:
        raise ValueError("凹槽最大面积必须是大于 0 的有限数值，单位为平方米")
    return {"fill_holes": fill, "max_hole_area_m2": float(area), "smooth_boundary": smooth,
            "ignore_small_boundaries": ignore, "max_gap_width_m": float(width),
            "fill_small_notches": notches, "max_notch_width_m": float(notch_width),
            "max_notch_area_m2": float(notch_area)}


def _ring_area_m2(ring, transform, crs):
    if crs is None:
        raise ValueError("原始影像缺少坐标系，无法按平方米计算去洞面积")
    cols, rows = np.asarray(ring, dtype=float).T
    xs, ys = transform * (cols, rows)
    lons, lats = project(crs, "EPSG:4326", xs.tolist(), ys.tolist())
    utm = utm_crs_for(float(np.mean(lons)), float(np.mean(lats)))
    xs, ys = project(crs, utm, xs.tolist(), ys.tolist())
    xy = np.column_stack([xs, ys])
    xy -= xy[0]  # 平移后计算，避免大地坐标的数值抵消。
    return abs(float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                            - xy[:, 1] * np.roll(xy[:, 0], -1)))) / 2


def _rasterize_regions(regions, shape, outer_only=False):
    geometries = [({"type": "Polygon", "coordinates": [r["hull"],
                   *([] if outer_only else r.get("holes", []))]}, 1) for r in regions]
    return rasterize(geometries, out_shape=shape, transform=Affine.identity(), dtype="uint8").astype(bool)


def _hole_is_protected(hole, forbidden, negative_points):
    if any(contains_point(hole, x, y) for x, y in negative_points):
        return True
    xy = np.asarray(hole)
    left, top = np.maximum(0, np.floor(xy.min(axis=0))).astype(int)
    right, bottom = np.minimum([forbidden.shape[1], forbidden.shape[0]], np.ceil(xy.max(axis=0))).astype(int)
    if right <= left or bottom <= top:
        return False
    tile = forbidden[top:bottom, left:right]
    if not tile.any():
        return False
    mask = rasterize([({"type": "Polygon", "coordinates": [hole]}, 1)],
                     out_shape=tile.shape, transform=Affine.translation(left, top), dtype="uint8")
    return bool(np.any(mask.astype(bool) & tile))


def _fill_small_notches(regions, transform, crs, coords, labels, forbidden, settings):
    """逐区域闭合窄槽口，按完整待填区域的地面面积筛选，保留原有洞。"""
    shape = forbidden.shape
    kernel = gap_kernel_for_width(settings["max_notch_width_m"], transform, crs, shape)
    if kernel == (1, 1):
        return regions
    base_all = _rasterize_regions(regions, shape)
    negative = [(int(x) + 0.5, int(y) + 0.5) for (x, y), label in zip(coords, labels) if label == 0]
    output = []

    def tile_for(geometry):
        xy = np.asarray(geometry["coordinates"][0])
        left, top = np.maximum(0, np.floor(xy.min(axis=0))).astype(int)
        right, bottom = np.minimum([shape[1], shape[0]], np.ceil(xy.max(axis=0))).astype(int)
        tile = rasterize([(geometry, 1)], out_shape=(bottom - top, right - left),
                         transform=Affine.translation(left, top), dtype="uint8").astype(bool)
        return np.s_[top:bottom, left:right], tile

    for region in regions:
        outer = _rasterize_regions([region], shape, outer_only=True)
        base = _rasterize_regions([region], shape)
        preserved_holes = outer & ~base
        closed = close_mask(outer, kernel)
        # 细槽口后方可能较宽；将新封闭的腔体与槽口作为整体计算面积。
        for geometry, _ in shapes(closed.astype("uint8"), mask=~closed, connectivity=4):
            xy = np.asarray(geometry["coordinates"][0])
            if (xy[:, 0] <= 0).any() or (xy[:, 0] >= shape[1]).any() or (xy[:, 1] <= 0).any() or (xy[:, 1] >= shape[0]).any():
                continue
            window, tile = tile_for(geometry)
            closed[window] |= tile
        additions = closed & ~outer
        if not additions.any():
            output.append(region)
            continue
        # 不能借修补把相邻的独立区域连接起来。
        other = base_all & ~base
        other = _binary_filter_axis(_binary_filter_axis(other, 3, 0, True), 3, 1, True)
        blocked = forbidden | other
        changed = False
        for geometry, _ in shapes(additions.astype("uint8"), mask=additions, connectivity=4):
            rings = geometry["coordinates"]
            area = _ring_area_m2(rings[0], transform, crs) - sum(_ring_area_m2(h, transform, crs) for h in rings[1:])
            if area > settings["max_notch_area_m2"] and not np.isclose(area, settings["max_notch_area_m2"], rtol=1e-8, atol=1e-10):
                continue
            if any(contains_point(rings[0], x, y) and not any(contains_point(h, x, y) for h in rings[1:]) for x, y in negative):
                continue
            window, tile = tile_for(geometry)
            if (blocked[window] & tile).any():
                continue
            outer[window] |= tile
            changed = True
        if changed:
            repaired = outer & ~preserved_holes
            output.extend(mask_to_regions(repaired, coords, labels))
        else:
            output.append(region)
    return output


def process_regions(regions, transform, crs, coords, labels, valid, excluded, settings):
    """按地面尺度去洞和修补凹槽，再平滑边界并恢复剩余洞与提示约束。"""
    if not regions or (not settings["fill_holes"] and not settings["smooth_boundary"] and not settings.get("fill_small_notches", False)):
        return regions
    forbidden = ~valid
    if excluded:
        forbidden = forbidden | _rasterize_regions(excluded, valid.shape)
    negative_points = [(int(x) + 0.5, int(y) + 0.5)
                       for (x, y), label in zip(coords, labels) if label == 0]
    processed = []
    for region in regions:
        holes = []
        for hole in region.get("holes", []):
            remove = False
            if settings["fill_holes"] and not _hole_is_protected(hole, forbidden, negative_points):
                area = _ring_area_m2(hole, transform, crs)
                maximum = settings["max_hole_area_m2"]
                remove = area < maximum and not np.isclose(area, maximum, rtol=1e-8, atol=1e-10)
            if not remove:
                holes.append(hole)
        processed.append({"hull": region["hull"], "holes": holes})
    if settings.get("fill_small_notches", False):
        processed = _fill_small_notches(processed, transform, crs, coords, labels, forbidden, settings)
    if not settings["smooth_boundary"]:
        return processed
    base = _rasterize_regions(processed, valid.shape)
    outer = _rasterize_regions(processed, valid.shape, outer_only=True)
    holes = outer & ~base
    # 三个原图像素尺度的平滑，仅处理外轮廓；洞的去留由面积选项决定。
    smoothed = np.asarray(Image.fromarray(outer.astype(np.uint8) * 255)
                          .filter(ImageFilter.GaussianBlur(radius=3))) >= 128
    smoothed &= ~holes & ~forbidden
    for (x, y), label in zip(coords, labels):
        ix, iy = int(x), int(y)
        if smoothed[iy, ix] != bool(label):
            # 提示点靠近边界时保留其原有局部轮廓，避免产生孤立的单像元区域。
            left, right = max(0, ix - 3), min(valid.shape[1], ix + 4)
            top, bottom = max(0, iy - 3), min(valid.shape[0], iy + 4)
            smoothed[top:bottom, left:right] = base[top:bottom, left:right] & ~forbidden[top:bottom, left:right]
    return mask_to_regions(smoothed, coords, labels)
