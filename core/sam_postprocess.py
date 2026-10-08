"""点提示区域的小洞填补与外边界平滑，面积单位为平方米。"""

import numpy as np
from affine import Affine
from PIL import Image, ImageFilter
from rasterio.features import rasterize
from rasterio.warp import transform as project

from core.prescription import utm_crs_for
from core.sam_roi import contains_point, mask_to_regions


def validate_postprocess(settings):
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("区域处理选项必须是对象")
    fill = settings.get("fill_holes", False)
    smooth = settings.get("smooth_boundary", False)
    ignore = settings.get("ignore_small_boundaries", False)
    if not all(isinstance(value, bool) for value in (fill, smooth, ignore)):
        raise ValueError("去洞、平滑边界和忽略细小分隔选项必须是布尔值")
    area = settings.get("max_hole_area_m2", 10.0)
    if isinstance(area, bool) or not isinstance(area, (int, float)) or not np.isfinite(area) or area <= 0:
        raise ValueError("去洞最大面积必须是大于 0 的有限数值，单位为平方米")
    width = settings.get("max_gap_width_m", 0.2)
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not np.isfinite(width) or not 0 < width <= 5:
        raise ValueError("细小分隔宽度阈值必须大于 0 且不超过 5 米")
    return {"fill_holes": fill, "max_hole_area_m2": float(area), "smooth_boundary": smooth,
            "ignore_small_boundaries": ignore, "max_gap_width_m": float(width)}


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


def process_regions(regions, transform, crs, coords, labels, valid, excluded, settings):
    """去洞按原图地面面积筛选；平滑外边界后恢复剩余洞与提示约束。"""
    if not settings["fill_holes"] and not settings["smooth_boundary"]:
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
