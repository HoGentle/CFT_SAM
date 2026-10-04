"""施肥处方图生成（移植自原 prescription.py，保持大疆智图兼容输出格式）。

输出格式要点（依据 debug_prescription.md 的设备兼容性排查结论）：
* 单波段 float32，无效像元写 NaN，不写 NoData 标签
* strip 存储（非 tile）、deflate 压缩
* 不嵌入 CRS / GeoKey / GDALMetadata 等扩展标签
* 同名 .tfw 世界文件承载空间参考

处方图写出后自动做地面栅格重采样（替代原网页的像素计数池化）：
用 rasterio.warp.reproject（与 gdalwarp 同源的 GDAL 变换内核）把结果
聚合到固定 1m × 1m 地面栅格（average，像元面积加权平均聚合），像元
尺寸按影像坐标系换算为真实地理地面距离，与影像原始地面分辨率无关。
"""

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import reproject

from core.diagnose import (
    build_compatible_gtiff_profile,
    make_unique_output_path,
    write_world_file,
)

RESAMPLING_METHODS = {
    "average": Resampling.average,
    "nearest": Resampling.nearest,
}


def build_formula_mapping(levels, base_level, base_fertilizer, change_coefficient, round_digits=2):
    """Q_i = Q_b * [1 + (b - i) * k]，按当前生育期的分级数量生成映射。"""
    mapping = {}
    for level in levels:
        value = float(base_fertilizer) * (1 + (int(base_level) - int(level)) * float(change_coefficient))
        mapping[str(int(level))] = round(float(value), int(round_digits))
    return mapping


def resolve_mapping(stage_levels, run_params, prescription_config, round_digits=2):
    """run_params: {"mode": "formula"|"manual", "formula": {...}, "manual": {...}}"""
    mode = run_params.get("mode", "formula")
    if mode == "manual":
        manual = run_params.get("manual") or {}
        mapping = {}
        for level in stage_levels:
            raw = manual.get(str(level), manual.get(int(level)))
            if raw is None:
                raise ValueError(f"手动映射缺少第 {level} 级的施肥量。")
            mapping[str(level)] = round(float(raw), round_digits)
        return mapping, "manual"

    formula_cfg = dict(prescription_config["formula_mode"])
    formula_cfg.update(run_params.get("formula") or {})
    mapping = build_formula_mapping(
        levels=stage_levels,
        base_level=formula_cfg["base_level"],
        base_fertilizer=formula_cfg["base_fertilizer"],
        change_coefficient=formula_cfg["change_coefficient"],
        round_digits=round_digits,
    )
    return mapping, "formula"


def write_prescription_raster(class_raster_path, output_path, mapping, defaults, progress=None):
    """按分块窗口读取诊断分级图，逐块写出大疆兼容处方图。"""
    class_raster_path = Path(class_raster_path)
    output_path = make_unique_output_path(Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_dtype = defaults.get("output_dtype", "float32")
    round_digits = int(defaults.get("round_digits", 2))
    block_size = max(64, int(defaults.get("processing_block_size", 1024)))
    input_nodata_values = defaults.get("input_nodata_values", [255])

    level_to_fert = {}
    for level_text, fert_value in mapping.items():
        try:
            level_to_fert[int(level_text)] = float(round(float(fert_value), round_digits))
        except (TypeError, ValueError):
            continue

    with rasterio.open(class_raster_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        src_nodata = src.nodata
        height, width = int(src.height), int(src.width)
        total_windows = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)

        out_profile = build_compatible_gtiff_profile(
            src_profile=profile,
            dtype=output_dtype,
            count=1,
            include_crs=False,
            nodata_value=None,
        )

        class_counts = {}
        unmatched_counts = {}
        unmatched_values_set = set()
        windows_done = 0

        with rasterio.open(output_path, "w", **out_profile) as dst:
            for row_off in range(0, height, block_size):
                row_h = min(block_size, height - row_off)
                for col_off in range(0, width, block_size):
                    col_w = min(block_size, width - col_off)
                    window = Window(col_off, row_off, col_w, row_h)
                    data = src.read(1, window=window)

                    valid_mask = src.dataset_mask(window=window) > 0
                    valid_mask &= np.isfinite(data)
                    if src_nodata is not None:
                        valid_mask &= data != src_nodata
                    for nodata_value in input_nodata_values:
                        valid_mask &= data != nodata_value

                    output_data = np.full(data.shape, np.nan, dtype=np.float32)
                    for level, fert in level_to_fert.items():
                        level_mask = valid_mask & (data == level)
                        if np.any(level_mask):
                            output_data[level_mask] = np.float32(fert)
                            key = str(level)
                            class_counts[key] = class_counts.get(key, 0) + int(level_mask.sum())

                    unmatched_mask = valid_mask & np.isnan(output_data)
                    if np.any(unmatched_mask):
                        uvals, ucnts = np.unique(data[unmatched_mask], return_counts=True)
                        for uv, uc in zip(uvals.tolist(), ucnts.tolist()):
                            ukey = str(int(uv))
                            unmatched_counts[ukey] = unmatched_counts.get(ukey, 0) + int(uc)
                            unmatched_values_set.add(int(uv))

                    if output_dtype == "float32":
                        dst.write(output_data.astype(np.float32), window=window, indexes=1)
                    elif output_dtype == "float64":
                        dst.write(output_data.astype(np.float64), window=window, indexes=1)
                    else:
                        dst.write(np.rint(output_data).astype(output_dtype), window=window, indexes=1)

                    windows_done += 1
                    if progress and windows_done % 10 == 0:
                        progress(windows_done / total_windows, f"写出处方图 {windows_done}/{total_windows} 块")

    world_file = write_world_file(output_path, transform)

    metadata = {
        "input_file": str(class_raster_path),
        "output_file": str(output_path),
        "world_file": str(world_file),
        "mapping": mapping,
        "class_counts": class_counts,
        "unmatched_class_counts": unmatched_counts,
        "unmatched_classes_written_as_nodata": sorted(unmatched_values_set),
        "output_dtype": output_dtype,
        "output_nodata_value": "NaN",
        "invalid_value_representation": "NaN",
        "unit": "kg/mu",
        "format_notes": {
            "compression": "deflate",
            "tiled": False,
            "embedded_crs": False,
            "nodata_tag_written": False,
            "processing": "tiled",
            "processing_block_size": block_size,
        },
    }
    metadata_path = make_unique_output_path(output_path.with_suffix(".json"))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "prescription_raster": output_path,
        "world_file": world_file,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "class_counts": class_counts,
        "unmatched_class_counts": unmatched_counts,
    }


def utm_crs_for(lon, lat):
    """选取覆盖该经纬度点的 WGS84 UTM 分区（南纬用 327xx 带）。"""
    zone = min(60, max(1, int(math.floor((lon + 180.0) / 6.0)) + 1))
    return CRS.from_epsg(32600 + zone if lat >= 0 else 32700 + zone)


def ground_meters_per_unit(crs, center_x, center_y):
    """测量源 CRS 中，中心点附近沿 x/y 每移动 1 个 CRS 单位对应的真实地面米数。

    统一用「源 CRS -> 本地 UTM 分区」量测基线的地面距离：经纬度坐标系按
    该纬度的真实米/度换算，投影坐标系（含非米单位）按真实地面距离换算，
    保证目标栅格尺寸始终基于地理地面距离而非像素计数。

    基线先用 0.01 单位粗测数量级，再用约 500m 地面基线精测：米制投影
    坐标若基线过短，坐标转换往返的浮点误差会显著放大测量结果。
    """
    if crs is None:
        raise ValueError(
            "影像缺少坐标系（CRS）信息，无法按地理地面距离重采样；请使用带地理参考的影像。"
        )
    wgs84 = CRS.from_epsg(4326)
    center_lon, center_lat = rasterio.warp.transform(crs, wgs84, [center_x], [center_y])
    utm_crs = utm_crs_for(float(center_lon[0]), float(center_lat[0]))

    def measure(delta_x, delta_y):
        xs, ys = rasterio.warp.transform(
            crs,
            utm_crs,
            [center_x, center_x + delta_x, center_x, center_x],
            [center_y, center_y, center_y, center_y + delta_y],
        )
        return (
            math.hypot(xs[1] - xs[0], ys[1] - ys[0]) / delta_x,
            math.hypot(xs[3] - xs[2], ys[3] - ys[2]) / delta_y,
        )

    rough_x, rough_y = measure(0.01, 0.01)
    delta_x = 500.0 / rough_x if rough_x > 0 else 0.01
    delta_y = 500.0 / rough_y if rough_y > 0 else 0.01
    m_per_unit_x, m_per_unit_y = measure(delta_x, delta_y)
    if not (m_per_unit_x > 0.0 and m_per_unit_y > 0.0):
        raise ValueError("无法根据影像坐标系计算地面分辨率，请检查影像地理参考。")
    return m_per_unit_x, m_per_unit_y


def _resolution_suffix(target_m):
    """目标地面分辨率的文件名后缀：1.0 -> "1m"，0.5 -> "0.5m"。"""
    return f"{target_m:g}m"


def write_resampled_prescription(
    input_path,
    output_path,
    src_crs,
    target_ground_resolution_m=1.0,
    method="average",
    progress=None,
):
    """把处方图自动重采样到固定 N m × N m 地面栅格（rasterio.warp 聚合）。

    目标像元尺寸由真实地面距离换算成源 CRS 单位（地理坐标下按中心纬度的
    米/度换算），源影像无论 GSD 是几厘米还是几米，输出栅格都是固定的
    1m 地面网格。average 聚合取各输出像元覆盖范围内有效源像元的平均，
    无效像元（NaN）经 src_nodata 声明后不参与聚合。

    输出保持大疆兼容格式：无内嵌 CRS、.tfw 世界文件承载参考、无效=NaN。
    """
    if method not in RESAMPLING_METHODS:
        raise ValueError(f"不支持的重采样方法: {method!r}（仅支持 average/nearest）")
    target_m = float(target_ground_resolution_m)
    if target_m <= 0:
        raise ValueError("target_ground_resolution_m 必须 > 0。")

    input_path = Path(input_path)
    output_path = make_unique_output_path(
        Path(output_path).with_name(
            f"{Path(output_path).stem}_{_resolution_suffix(target_m)}{Path(output_path).suffix}"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(0.05, f"正在计算 {target_m:g}m 地面栅格...")
    with rasterio.open(input_path) as src:
        src_transform = src.transform
        src_crs = src.crs or src_crs
        width, height = int(src.width), int(src.height)

        center_x = src_transform.c + src_transform.a * (width / 2.0) + src_transform.b * (height / 2.0)
        center_y = src_transform.f + src_transform.d * (width / 2.0) + src_transform.e * (height / 2.0)
        m_per_unit_x, m_per_unit_y = ground_meters_per_unit(src_crs, center_x, center_y)

        src_gsd_x = math.hypot(src_transform.a, src_transform.d) * m_per_unit_x
        src_gsd_y = math.hypot(src_transform.b, src_transform.e) * m_per_unit_y

        # 目标像元尺寸（源 CRS 单位）与输出网格：以源左上角为原点，范围向上取整
        target_units_x = target_m / m_per_unit_x
        target_units_y = target_m / m_per_unit_y
        scale_x = target_units_x / math.hypot(src_transform.a, src_transform.d)
        scale_y = target_units_y / math.hypot(src_transform.b, src_transform.e)
        out_w = max(1, int(math.ceil(width / scale_x)))
        out_h = max(1, int(math.ceil(height / scale_y)))
        if out_w * out_h > 250_000_000:
            raise ValueError(
                f"{target_m:g}m 地面栅格输出像元数过大，请检查影像范围是否异常。"
            )

        out_transform = src_transform * Affine(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0)

        out_profile = build_compatible_gtiff_profile(
            src_profile=src.profile.copy(),
            dtype="float32",
            count=1,
            include_crs=False,
            nodata_value=None,
        )
        out_profile.update(width=out_w, height=out_h, transform=out_transform)

        if progress:
            progress(0.3, f"正在按 {target_m:g}m × {target_m:g}m 地面栅格聚合重采样...")
        destination = np.full((out_h, out_w), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=out_transform,
            dst_crs=src_crs,
            dst_nodata=np.nan,
            resampling=RESAMPLING_METHODS[method],
        )
        if progress:
            progress(0.85, "正在写出重采样处方图...")
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(destination, indexes=1)

        # 统计可视化数据：每个输出像元地面面积 = target_m × target_m（构造保证）
        valid_mask = np.isfinite(destination)
        valid_cells = int(valid_mask.sum())
        cell_area_m2 = target_m * target_m
        area_m2 = valid_cells * cell_area_m2
        if valid_cells:
            amount_sum = float(destination[valid_mask].sum(dtype=np.float64))  # Σ kg/亩
            total_fertilizer_kg = amount_sum * cell_area_m2 / 666.6666666666666
            avg_fertilizer_kg_per_mu = amount_sum / valid_cells
        else:
            total_fertilizer_kg = 0.0
            avg_fertilizer_kg_per_mu = 0.0
        statistics = {
            "valid_cells": valid_cells,
            "grid_cell_m": target_m,
            "area_m2": round(area_m2, 2),
            "area_mu": round(area_m2 / 666.6666666666666, 4),
            "total_fertilizer_kg": round(total_fertilizer_kg, 4),
            "avg_fertilizer_kg_per_mu": round(avg_fertilizer_kg_per_mu, 4),
        }

    world_file = write_world_file(output_path, out_transform)

    metadata = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "world_file": str(world_file),
        "resampling": {
            "mode": "ground_distance_grid",
            "engine": "rasterio.warp.reproject (GDAL)",
            "method": method,
            "target_ground_resolution_m": target_m,
            "source_ground_resolution_m": {"x": round(src_gsd_x, 4), "y": round(src_gsd_y, 4)},
            "meters_per_crs_unit": {"x": m_per_unit_x, "y": m_per_unit_y},
            "source_crs": src_crs.to_string(),
            "source_grid_size": {"width": width, "height": height},
            "output_grid_size": {"width": out_w, "height": out_h},
            "aggregation": (
                "average 像元面积加权平均聚合（NaN 不参与统计）"
                if method == "average"
                else "最近邻取样（NaN 视为无效）"
            ),
        },
        "statistics": statistics,
        "unit": "kg/mu",
        "output_dtype": "float32",
        "output_nodata_value": "NaN",
        "invalid_value_representation": "NaN",
        "format_notes": {
            "compression": "deflate",
            "tiled": False,
            "embedded_crs": False,
            "nodata_tag_written": False,
            "processing": "ground_grid_resample",
        },
    }
    metadata_path = make_unique_output_path(output_path.with_suffix(".json"))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if progress:
        progress(1.0, "重采样完成")
    return {
        "prescription_raster": output_path,
        "world_file": world_file,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "statistics": statistics,
    }
