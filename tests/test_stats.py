"""core.stats 的面积统计测试。

* 经纬度多边形地面面积（与 WGS84 弧长级数独立对照）
* 栅格有效像元面积（dataset_mask 掩膜 + 地面距离换算）

用法: python tests/test_stats.py
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REDO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REDO_ROOT))

# 先导入 core（其 __init__ 会在 rasterio 导入前修正 Windows 下的 proj.db 冲突）
from core.stats import lonlat_polygons_area, raster_valid_area  # noqa: E402

import rasterio  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import Affine  # noqa: E402


def wgs84_meters_per_degree(lat_deg):
    lat = math.radians(lat_deg)
    m_lat = 111132.954 - 559.822 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat) - 0.0023 * math.cos(6 * lat)
    m_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat) + 0.118 * math.cos(5 * lat)
    return m_lon, m_lat


def test_polygon_area():
    """(113.5, 30.5) 处 100m × 60m 矩形与两个区域求和。"""
    lon0, lat0 = 113.5, 30.5
    m_lon, m_lat = wgs84_meters_per_degree(lat0)
    d_lon, d_lat = 100.0 / m_lon, 60.0 / m_lat
    rect = [[lon0, lat0], [lon0 + d_lon, lat0], [lon0 + d_lon, lat0 + d_lat], [lon0, lat0 + d_lat]]

    area = lonlat_polygons_area([rect])
    assert abs(area["area_m2"] - 6000.0) < 30.0, f"矩形面积偏差过大: {area}"
    assert abs(area["area_mu"] - area["area_m2"] * 3.0 / 2000.0) < 1e-4

    two = lonlat_polygons_area([rect, rect])
    assert abs(two["area_m2"] - 12000.0) < 60.0

    assert lonlat_polygons_area([])["area_m2"] == 0.0
    assert lonlat_polygons_area([[[0, 0], [1, 1]]])["area_m2"] == 0.0  # 顶点不足
    print("[通过] 经纬度多边形面积")


def test_raster_valid_area(tmp):
    """带 nodata 掩膜的经纬度栅格：有效像元数 × 像元地面面积。"""
    lon0, lat0 = 113.5, 30.5
    m_lon, m_lat = wgs84_meters_per_degree(lat0)
    gsd_x_deg, gsd_y_deg = 0.5 / m_lon, 0.5 / m_lat  # 0.5m GSD
    data = np.full((200, 100), 5.0, dtype="float32")
    data[100:, :] = 255.0  # 下半幅无效
    path = tmp / "band.tif"
    profile = {
        "driver": "GTiff", "width": 100, "height": 200, "count": 1,
        "dtype": "float32", "crs": CRS.from_epsg(4326), "nodata": 255.0,
        "transform": Affine(gsd_x_deg, 0, lon0, 0, -gsd_y_deg, lat0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, indexes=1)

    area = raster_valid_area(path)
    assert area["valid_pixels"] == 100 * 100, f"有效像元数错误: {area}"
    expected_m2 = 10000 * 0.5 * 0.5
    assert abs(area["area_m2"] - expected_m2) < expected_m2 * 0.01, \
        f"面积偏差过大: {area} vs {expected_m2}"
    assert abs(area["pixel_size_m"]["x"] - 0.5) < 0.005
    print("[通过] 栅格有效面积统计")


def main():
    with tempfile.TemporaryDirectory(prefix="redo_stats_") as tmp_name:
        test_polygon_area()
        test_raster_valid_area(Path(tmp_name))
    print("\n面积统计测试全部通过。")


if __name__ == "__main__":
    main()
