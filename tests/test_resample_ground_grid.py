"""write_resampled_prescription 的地面栅格重采样测试（目标分辨率由配置驱动）。

不依赖外部数据：合成带地理参考的处方图，验证
* 目标像元地面尺寸为固定 N m（经纬度源按纬度换算、投影源按米直接换算）
* average 聚合语义（有效像元面积加权平均，NaN 不参与、空洞内部输出 NaN）
* 大疆兼容输出格式（无内嵌 CRS、.tfw 世界文件、float32、无效=NaN）
* 缺少 CRS 时给出明确报错

经纬度源验证 5m / 2m / 1m 三档目标分辨率（1m 为当前默认配置）。
用法: python tests/test_resample_ground_grid.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REDO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REDO_ROOT))

# 先导入 core（其 __init__ 会在 rasterio 导入前修正 Windows 下的 proj.db 冲突）
from core.prescription import write_resampled_prescription  # noqa: E402

import rasterio  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import Affine  # noqa: E402


def wgs84_meters_per_degree(lat_deg):
    """WGS84 弧长级数（独立于实现内的 UTM 量测，用于交叉验证）。"""
    lat = math.radians(lat_deg)
    m_lat = 111132.954 - 559.822 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat) - 0.0023 * math.cos(6 * lat)
    m_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat) + 0.118 * math.cos(5 * lat)
    return m_lon, m_lat


def write_synthetic(path, transform, crs, data):
    profile = {
        "driver": "GTiff", "width": data.shape[1], "height": data.shape[0],
        "count": 1, "dtype": "float32", "transform": transform,
    }
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), indexes=1)


def read_tif(path):
    with rasterio.open(path) as src:
        attrs = {
            "transform": src.transform,
            "width": int(src.width),
            "height": int(src.height),
            "crs": src.crs,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }
        return src.read(1), attrs


def check_dji_format(out_path, expected_target_m):
    tfw = Path(str(out_path).replace(".tif", ".tfw"))
    assert tfw.is_file(), "缺少 .tfw 世界文件"
    _, attrs = read_tif(out_path)
    assert attrs["crs"] is None, "处方图不应内嵌 CRS"
    assert attrs["nodata"] is None, "处方图不应写 nodata 标签"
    assert attrs["dtype"] == "float32"
    meta_path = Path(str(out_path).replace(".tif", ".json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["resampling"]["mode"] == "ground_distance_grid"
    assert meta["resampling"]["method"] == "average"
    assert meta["resampling"]["target_ground_resolution_m"] == expected_target_m
    return meta


def test_geographic_source(tmp):
    """经纬度源（EPSG:4326，约 1m GSD）：5m / 2m / 1m 三档目标栅格均正确。"""
    lon0, lat0 = 113.5, 30.5
    m_lon, m_lat = wgs84_meters_per_degree(lat0)
    gsd_x_deg, gsd_y_deg = 1.0 / m_lon, 1.0 / m_lat
    data = np.full((150, 200), 10.0, dtype="float32")
    data[:, 100:] = 20.0
    data[50:70, 30:50] = np.nan  # 约 20m × 20m 的无效像元块
    src_path = tmp / "geo_prescription.tif"
    write_synthetic(src_path, Affine(gsd_x_deg, 0, lon0, 0, -gsd_y_deg, lat0), CRS.from_epsg(4326), data)

    for target_m in (5.0, 2.0, 1.0):
        result = write_resampled_prescription(
            input_path=src_path,
            output_path=tmp / f"geo_out_{target_m:g}m.tif",
            src_crs=CRS.from_epsg(4326),
            target_ground_resolution_m=target_m,
            method="average",
        )
        out_path = result["prescription_raster"]
        assert f"_{target_m:g}m.tif" in out_path.name
        out, attrs = read_tif(out_path)

        # 输出像元地面尺寸 = 目标值（与实现内的 UTM 量测方式相互独立）
        tf = attrs["transform"]
        center_lat = tf.f + tf.e * attrs["height"] / 2.0
        m_lon_c, m_lat_c = wgs84_meters_per_degree(center_lat)
        gsd_x_m = tf.a * m_lon_c
        gsd_y_m = abs(tf.e) * m_lat_c
        tol = max(0.05, target_m * 0.01)
        assert abs(gsd_x_m - target_m) < tol, f"{target_m}m: 东西向像元 {gsd_x_m:.4f}m 偏差过大"
        assert abs(gsd_y_m - target_m) < tol, f"{target_m}m: 南北向像元 {gsd_y_m:.4f}m 偏差过大"

        # 输出范围覆盖整个源（误差在一个像元内）
        extent_x_m = 200 * gsd_x_deg * m_lon
        extent_y_m = 150 * gsd_y_deg * m_lat
        assert abs(attrs["width"] * gsd_x_m - extent_x_m) <= gsd_x_m + 0.01
        assert abs(attrs["height"] * gsd_y_m - extent_y_m) <= gsd_y_m + 0.01

        # average 聚合：左半 10、右半 20（取远离边界与空洞的内点）
        scale_x = tf.a / gsd_x_deg
        scale_y = abs(tf.e) / gsd_y_deg
        left = out[int(75 / scale_y), int(20 / scale_x)]
        right = out[int(75 / scale_y), int(125 / scale_x)]
        assert np.allclose(left, 10.0, atol=1e-4), f"{target_m}m: 左半均值错误: {left}"
        assert np.allclose(right, 20.0, atol=1e-4), f"{target_m}m: 右半均值错误: {right}"

        # 空洞内部格输出 NaN（空洞地面范围约 [50,70) × [30,50) m，取 [55,65) × [35,45) 内点）
        assert np.isnan(out[int(55 / target_m):int(65 / target_m),
                            int(35 / target_m):int(45 / target_m)]).all(), \
            f"{target_m}m: NaN 源块内部应输出 NaN"
        assert np.isfinite(out).any(), f"{target_m}m: 输出不应全为 NaN"

        check_dji_format(out_path, target_m)
        print(f"[通过] 经纬度源 {target_m:g}m 地面栅格重采样 + average 聚合")


def test_projected_source(tmp):
    """投影源（UTM，米制，0.25m GSD）：输出像元应精确 1.0m，聚合正确。"""
    data = np.full((150, 200), 7.5, dtype="float32")
    data[0:40, :] = np.nan  # 顶部 10m 无效带
    src_path = tmp / "utm_prescription.tif"
    write_synthetic(
        src_path, Affine(0.25, 0, 500000.0, 0, -0.25, 3400000.0), CRS.from_epsg(32649), data
    )

    result = write_resampled_prescription(
        input_path=src_path,
        output_path=tmp / "utm_out.tif",
        src_crs=CRS.from_epsg(32649),
        target_ground_resolution_m=1.0,
        method="average",
    )
    out_path = result["prescription_raster"]
    out, attrs = read_tif(out_path)

    assert abs(attrs["transform"].a - 1.0) < 1e-5, f"UTM 源输出像元宽 {attrs['transform'].a} ≠ 1m"
    assert abs(abs(attrs["transform"].e) - 1.0) < 1e-5, \
        f"UTM 源输出像元高 {abs(attrs['transform'].e)} ≠ 1m"
    assert np.allclose(out[30, 25], 7.5, atol=1e-4)
    assert np.isnan(out[0:8, :]).all(), "源顶部 NaN 带内部应输出 NaN"
    check_dji_format(out_path, 1.0)
    print("[通过] UTM 投影源精确 1m 栅格重采样")


def test_missing_crs(tmp):
    """缺少 CRS 时应报出明确错误而不是产出错误栅格。"""
    data = np.full((20, 20), 1.0, dtype="float32")
    src_path = tmp / "nocrs_prescription.tif"
    write_synthetic(src_path, Affine(2.0, 0, 0.0, 0, -2.0, 0.0), None, data)
    try:
        write_resampled_prescription(
            input_path=src_path, output_path=tmp / "nocrs_out.tif", src_crs=None
        )
    except ValueError as exc:
        assert "CRS" in str(exc) or "坐标系" in str(exc), f"报错信息不明确: {exc}"
        print("[通过] 缺少 CRS 时给出明确报错")
        return
    raise AssertionError("缺少 CRS 时未报错")


def main():
    with tempfile.TemporaryDirectory(prefix="redo_resample_") as tmp_name:
        tmp = Path(tmp_name)
        test_geographic_source(tmp)
        test_projected_source(tmp)
        test_missing_crs(tmp)
    print("\n地面栅格重采样测试全部通过。")


if __name__ == "__main__":
    main()
