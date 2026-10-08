"""面积与区域统计（预览图下方的可视化数据来源）。

* 影像有效面积：按 dataset_mask（alpha / nodata 标签）统计有效像元，
  像元地面尺寸与重采样共用同一套「源 CRS -> 本地 UTM」地面距离换算。
* 经纬度多边形面积：顶点变换到 UTM 后用鞋匠公式求和（田块尺度下
  UTM 平面近似误差可忽略）。
面积单位同时给出 m² 与亩（1 亩 = 2000/3 m²）。
"""

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from core.prescription import ground_meters_per_unit, utm_crs_for

MU_PER_SQM = 3.0 / 2000.0  # m² -> 亩


def _mu(area_m2):
    return area_m2 * MU_PER_SQM


def raster_valid_area(path):
    """统计单波段栅格的有效像元数与地面面积。

    返回 {"valid_pixels", "pixel_size_m", "area_m2", "area_mu"}。
    """
    path = Path(path)
    with rasterio.open(path) as src:
        transform = src.transform
        width, height = int(src.width), int(src.height)
        center_x = transform.c + transform.a * (width / 2.0) + transform.b * (height / 2.0)
        center_y = transform.f + transform.d * (width / 2.0) + transform.e * (height / 2.0)
        mpu_x, mpu_y = ground_meters_per_unit(src.crs, center_x, center_y)
        pixel_w_m = math.hypot(transform.a, transform.d) * mpu_x
        pixel_h_m = math.hypot(transform.b, transform.e) * mpu_y
        pixel_area_m2 = pixel_w_m * pixel_h_m

        block = 1024
        valid_pixels = 0
        for row0 in range(0, height, block):
            row_h = min(block, height - row0)
            for col0 in range(0, width, block):
                col_w = min(block, width - col0)
                window = Window(col0, row0, col_w, row_h)
                valid_pixels += int(np.count_nonzero(src.dataset_mask(window=window) > 0))

    area_m2 = valid_pixels * pixel_area_m2
    area_m2 = round(area_m2, 2)
    return {
        "valid_pixels": valid_pixels,
        "pixel_size_m": {"x": round(pixel_w_m, 4), "y": round(pixel_h_m, 4)},
        "area_m2": area_m2,
        "area_mu": round(area_m2 * MU_PER_SQM, 4),
    }


def lonlat_polygons_area(polygons_lonlat):
    """经纬度多边形列表的地面总面积（UTM 平面鞋匠公式）。

    支持顶点数组及包含 hull、holes 的区域字典；空洞面积从外轮廓扣除。
    """
    polygons_lonlat = [p for p in (polygons_lonlat or [])
                       if len(p["hull"] if isinstance(p, dict) else p) >= 3]
    if not polygons_lonlat:
        return {"area_m2": 0.0, "area_mu": 0.0}

    first = polygons_lonlat[0]
    if isinstance(first, dict):
        first = first["hull"]
    center_lon = sum(float(p[0]) for p in first) / len(first)
    center_lat = sum(float(p[1]) for p in first) / len(first)
    utm = utm_crs_for(center_lon, center_lat)

    total_m2 = 0.0

    def ring_area(poly):
        lons = [float(p[0]) for p in poly]
        lats = [float(p[1]) for p in poly]
        xs, ys = rasterio.warp.transform("EPSG:4326", utm, lons, lats)
        shoelace = 0.0
        for i in range(len(xs)):
            j = (i + 1) % len(xs)
            shoelace += xs[i] * ys[j] - xs[j] * ys[i]
        return abs(shoelace) / 2.0

    for poly in polygons_lonlat:
        if isinstance(poly, dict):
            total_m2 += max(0.0, ring_area(poly["hull"]) - sum(ring_area(h) for h in poly.get("holes", [])))
        else:
            total_m2 += ring_area(poly)

    total_m2 = round(total_m2, 2)
    return {"area_m2": total_m2, "area_mu": round(total_m2 * MU_PER_SQM, 4)}
