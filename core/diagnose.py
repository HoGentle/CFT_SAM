"""营养诊断：两遍分块流式处理（移植自原 diagnose.py）。

输入为 4 个单波段通道文件（B1 RedEdge / B2 Green / B3 Red / B4 NIR），
诊断模型使用 B2/B3/B4（当前 8 个生育期模型均由绿光/红光/近红外构成）；
B1 参与完整性校验，但模型公式未直接使用。

相对原实现的行为差异（有意修复）：
* 分位数样本改为全图均匀抽样（蓄水池采样，固定随机种子）。
  原实现在达到 max_quantile_samples 上限时直接停止遍历，导致样本
  集中在图像左上角，分级阈值存在空间偏置，与代码注释声称的
  "随机抽样"不符。
"""

import contextlib
import json
from pathlib import Path

import numpy as np
import rasterio

from core.roi import rasterize_polygons_in_window

PROFILE_KEYS_TO_DROP = {
    "blockxsize",
    "blockysize",
    "compress",
    "interleave",
    "nodata",
    "photometric",
    "tiled",
}

STAGE_NAME_SUFFIX = {
    "seedling": "youmiao",
    "tillering": "fennie",
    "jointing": "bajie",
    "booting": "yunsui",
    "heading": "chousui",
    "flowering": "yanghua",
    "pre_maturity": "chengshuqian",
    "maturity": "chengshu",
}


def safe_divide(a, b, eps=1e-6):
    return a / np.maximum(b, eps)


def compute_index(stage_key, bands):
    b2 = bands["B2"]
    b3 = bands["B3"]
    b4 = bands["B4"]

    if stage_key == "seedling":
        return (1.0 + 0.10) * safe_divide(b4 - b3, b4 + b3 + 0.1)
    if stage_key == "tillering":
        return (1.0 + 0.16) * safe_divide(b4 - b3, b4 + b3 + 0.16)
    if stage_key == "jointing":
        return (1.0 + 0.16) * safe_divide(b4 - b3, b4 + b3 + 0.16)
    if stage_key == "booting":
        return safe_divide(b4, b3)
    if stage_key == "heading":
        return safe_divide(b4 - b3, b4 + b3)
    if stage_key == "flowering":
        return safe_divide(b2 - b3, b2 + b3)
    if stage_key == "pre_maturity":
        return b4 - b3
    if stage_key == "maturity":
        return safe_divide(b4, b3)
    raise ValueError(f"不支持的生育期: {stage_key}")


def compute_diagnosis(stage_key, index_array):
    if stage_key == "seedling":
        return 95.7 * index_array + 18.4
    if stage_key == "tillering":
        return 2845.6 * index_array ** 2 - 2953.2 * index_array + 987.5
    if stage_key == "jointing":
        return np.power(26.107, 2.6114 * index_array)
    if stage_key == "booting":
        return -0.7375 * index_array + 9.6073
    if stage_key == "heading":
        return 2.8339 * np.exp(1.3655 * index_array)
    if stage_key == "flowering":
        return -7.354 * index_array + 7.9181
    if stage_key == "pre_maturity":
        return 0.3878 * index_array - 195.83
    if stage_key == "maturity":
        return -48.304 * index_array ** 2 + 518.11 * index_array - 528.44
    raise ValueError(f"不支持的生育期: {stage_key}")


def resolve_quantile_thresholds(samples, thresholds):
    """按分位数断点解析各级 [min, max) 区间（min 开区间、max 闭区间）。"""
    if samples is None or samples.size == 0:
        raise ValueError("没有任何有效像素样本，无法计算分级阈值。")

    resolved = []
    previous_quantile = 0.0
    previous_break_value = None

    for index, threshold in enumerate(thresholds):
        current_quantile = float(threshold["quantile"])
        if not 0.0 < current_quantile <= 1.0:
            raise ValueError(f"分位断点必须位于 (0, 1] 区间内: {current_quantile}")
        if current_quantile < previous_quantile:
            raise ValueError("分位断点必须按从小到大排序。")

        break_value = float(np.quantile(samples, current_quantile))
        is_last = index == len(thresholds) - 1
        resolved.append({
            "level": threshold["level"],
            "growth_label": threshold.get("growth_label"),
            "quantile": current_quantile,
            "min": previous_break_value,
            "max": None if is_last else break_value,
            "min_inclusive": previous_break_value is None,
            "max_inclusive": True,
        })
        previous_quantile = current_quantile
        previous_break_value = break_value

    if previous_quantile != 1.0:
        raise ValueError("最后一级的分位断点必须为 1.0。")
    return resolved


def classify_block(diagnosis_array, valid_mask, resolved_thresholds, nodata_value):
    class_map = np.full(diagnosis_array.shape, nodata_value, dtype=np.uint8)
    for threshold in resolved_thresholds:
        current_mask = valid_mask.copy()
        min_value = threshold.get("min")
        max_value = threshold.get("max")
        if min_value is not None:
            current_mask &= diagnosis_array > min_value
        if max_value is not None:
            current_mask &= diagnosis_array <= max_value
        class_map[current_mask] = threshold["level"]
    return class_map


def build_compatible_gtiff_profile(src_profile, dtype, count=1, include_crs=True, nodata_value=None):
    profile = src_profile.copy()
    for key in PROFILE_KEYS_TO_DROP:
        profile.pop(key, None)
    if not include_crs:
        profile.pop("crs", None)
    profile.update(
        driver="GTiff",
        dtype=dtype,
        count=count,
        compress="deflate",
        tiled=False,
        nodata=nodata_value,
    )
    return profile


def write_world_file(output_path, transform):
    world_file = Path(output_path).with_suffix(".tfw")
    lines = (
        f"{transform.a:.10f}",
        f"{transform.d:.10f}",
        f"{transform.b:.10f}",
        f"{transform.e:.10f}",
        f"{transform.c + (transform.a / 2.0) + (transform.b / 2.0):.10f}",
        f"{transform.f + (transform.d / 2.0) + (transform.e / 2.0):.10f}",
    )
    world_file.write_text("\n".join(lines) + "\n", encoding="ascii")
    return world_file


def make_unique_output_path(target_path):
    target_path = Path(target_path)
    if not target_path.exists():
        return target_path
    index = 1
    while True:
        candidate = target_path.parent / f"{target_path.stem}{index:02d}{target_path.suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _sample_block_values(samples_state, diagnosis_array, valid_mask):
    """分位数样本收集。

    * max_quantile_samples 关闭时：全量收集（与原实现一致，内存随有效像素增长）。
    * 开启时：蓄水池均匀抽样，每个有效值等概率进入固定容量样本集，
      全图任意区域被采到的概率一致（修复原实现样本偏向左上角的问题）。
    """
    values = diagnosis_array[valid_mask]
    n = int(values.size)
    if n == 0:
        return

    if samples_state["mode"] == "all":
        samples_state["chunks"].append(np.ascontiguousarray(values))
        samples_state["seen"] += n
        return

    reservoir = samples_state["reservoir"]
    capacity = reservoir.size
    seen = samples_state["seen"]

    if seen < capacity:
        take = min(capacity - seen, n)
        reservoir[seen:seen + take] = values[:take]
        samples_state["seen"] = seen + take
        if n <= take:
            return
        values = values[take:]
        n -= take
        seen = samples_state["seen"]

    global_index = np.arange(seen, seen + n, dtype=np.float64)
    rng = samples_state["rng"]
    accept = rng.random(n) < capacity / (global_index + 1.0)
    if np.any(accept):
        slots = rng.integers(0, capacity, size=int(np.count_nonzero(accept)))
        reservoir[slots] = values[accept]
    samples_state["seen"] = seen + n


def run_diagnosis(band_paths, stage_config, output_dir, defaults, progress=None, base_name=None,
                  roi_polygons=None, roi_meta=None):
    """执行两遍分块诊断，返回结果信息 dict。

    band_paths: {"B1": path, "B2": path, "B3": path, "B4": path}
    progress:   callable(fraction: 0..1, message: str)，可为 None
    base_name:  输出文件名基准（不含扩展名）；缺省由 B2 文件名派生
    roi_polygons: 诊断网格 (col,row) 像素坐标多边形列表；给出时仅对
                  区域内有效像素做诊断与统计，区域外输出无效值
    roi_meta:   附加到元数据的 ROI 信息（经纬度顶点等）
    """
    from core.inputs import iter_band_windows

    stage_key = stage_config["stage_key"]
    block_size = max(64, int(defaults.get("processing_block_size", 1024)))
    max_samples = defaults.get("max_quantile_samples")
    if max_samples in (None, "", 0, "0"):
        max_samples = None
    else:
        max_samples = int(max_samples)
    nodata_value = int(defaults.get("nodata_value", 255))
    save_value_raster = bool(defaults.get("save_value_raster", True))

    with rasterio.open(band_paths["B2"]) as src_b2:
        profile = src_b2.profile.copy()
        transform = src_b2.transform
        height, width = int(src_b2.height), int(src_b2.width)

    total_windows = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not base_name:
        base_name = Path(band_paths["B2"]).stem
        for suffix in ("_green",):
            if base_name.lower().endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break
    stage_suffix = STAGE_NAME_SUFFIX.get(stage_key, stage_key)

    class_output_path = make_unique_output_path(output_dir / f"{base_name}_diagnose_{stage_suffix}.tif")
    value_output_path = None
    if save_value_raster:
        value_output_path = make_unique_output_path(
            class_output_path.with_name(f"{class_output_path.stem}_value.tif")
        )

    # ---- 第一遍：抽样诊断值，计算分位阈值（仅 ROI 内有效像素） ----
    samples_state = {
        "rng": np.random.default_rng(20240101),
        "mode": "all" if max_samples is None else "reservoir",
        "chunks": [],
        "reservoir": np.zeros(max_samples or 0, dtype="float32"),
        "seen": 0,
    }
    windows_done = 0
    for window, bands, valid_mask in iter_band_windows(band_paths, height, width, block_size):
        valid_mask = valid_mask.copy()
        if roi_polygons is not None:
            valid_mask &= rasterize_polygons_in_window(roi_polygons, window)
        index_array = compute_index(stage_key, bands)
        diagnosis_array = compute_diagnosis(stage_key, index_array)
        valid = valid_mask & np.isfinite(index_array) & np.isfinite(diagnosis_array)
        _sample_block_values(samples_state, diagnosis_array, valid)
        windows_done += 1
        if progress and windows_done % 10 == 0:
            progress(windows_done / total_windows * 0.5, f"统计诊断值样本 {windows_done}/{total_windows} 块")

    if samples_state["mode"] == "all":
        samples = (
            np.concatenate(samples_state["chunks"])
            if samples_state["chunks"]
            else np.zeros(0, dtype="float32")
        )
    elif samples_state["seen"] <= max_samples:
        samples = samples_state["reservoir"][:samples_state["seen"]]
    else:
        samples = samples_state["reservoir"]
    if samples.size == 0:
        if roi_polygons is not None:
            raise ValueError("感兴趣区域内没有任何有效像素，请检查区域是否落在影像有效范围内。")
        raise ValueError("影像中没有任何有效像素，无法计算分级阈值。")

    resolved_thresholds = resolve_quantile_thresholds(samples, stage_config["thresholds"])
    if progress:
        progress(0.5, "分位阈值计算完成，开始写出诊断图")

    # ---- 第二遍：分块写出分类图与诊断值图 ----
    class_profile = build_compatible_gtiff_profile(
        profile, dtype="uint8", count=1, include_crs=True, nodata_value=nodata_value,
    )
    value_profile = None
    if value_output_path is not None:
        value_profile = build_compatible_gtiff_profile(
            profile, dtype="float32", count=1, include_crs=True, nodata_value=None,
        )

    class_counts = {}
    value_min = None
    value_max = None
    with contextlib.ExitStack() as stack:
        class_dst = stack.enter_context(
            rasterio.open(class_output_path, "w", **class_profile)
        )
        value_dst = None
        if value_output_path is not None:
            value_dst = stack.enter_context(
                rasterio.open(value_output_path, "w", **value_profile)
            )
        windows_done = 0
        for window, bands, valid_mask in iter_band_windows(band_paths, height, width, block_size):
            valid_mask = valid_mask.copy()
            if roi_polygons is not None:
                valid_mask &= rasterize_polygons_in_window(roi_polygons, window)
            index_array = compute_index(stage_key, bands)
            diagnosis_array = compute_diagnosis(stage_key, index_array)
            valid = valid_mask & np.isfinite(index_array) & np.isfinite(diagnosis_array)
            class_map = classify_block(diagnosis_array, valid, resolved_thresholds, nodata_value)
            class_dst.write(class_map, window=window, indexes=1)

            if valid.any():
                block_values = diagnosis_array[valid]
                block_min = float(block_values.min())
                block_max = float(block_values.max())
                value_min = block_min if value_min is None else min(value_min, block_min)
                value_max = block_max if value_max is None else max(value_max, block_max)

            valid_classes = class_map[valid]
            if valid_classes.size:
                uniq, counts = np.unique(valid_classes, return_counts=True)
                for level, count in zip(uniq.tolist(), counts.tolist()):
                    key = str(int(level))
                    class_counts[key] = class_counts.get(key, 0) + int(count)

            if value_output_path is not None:
                value_block = np.full(diagnosis_array.shape, np.nan, dtype="float32")
                value_block[valid] = diagnosis_array[valid].astype("float32")
                value_dst.write(value_block, window=window, indexes=1)

            windows_done += 1
            if progress and windows_done % 10 == 0:
                progress(0.5 + windows_done / total_windows * 0.5, f"写出诊断图 {windows_done}/{total_windows} 块")

    class_world_file = write_world_file(class_output_path, transform)
    value_world_file = write_world_file(value_output_path, transform) if value_output_path else None

    total_valid = sum(class_counts.values())
    metadata = {
        "input_channels": {role: str(path) for role, path in band_paths.items()},
        "source_type": "single_band_group",
        "make_mode": "roi" if roi_polygons is not None else "full",
        "stage_key": stage_key,
        "stage_name": stage_config["name"],
        "diagnosis_name": stage_config["diagnosis_name"],
        "index_name": stage_config["index_name"],
        "index_formula": stage_config["index_formula"],
        "diagnosis_formula": stage_config["diagnosis_formula"],
        "output_class_raster": str(class_output_path),
        "output_value_raster": str(value_output_path) if value_output_path else None,
        "class_counts": class_counts,
        "valid_pixel_count": total_valid,
        "value_range": {"min": value_min, "max": value_max},
        "level_quantiles": stage_config["thresholds"],
        "level_thresholds": resolved_thresholds,
        "world_file": str(class_world_file),
        "value_world_file": str(value_world_file) if value_world_file else None,
        "format_notes": {
            "compression": "deflate",
            "tiled": False,
            "class_raster_embedded_crs": True,
            "class_nodata": nodata_value,
            "value_raster_invalid_as": "NaN",
            "processing": "tiled_two_pass",
            "processing_block_size": block_size,
            "quantile_sampling": "uniform_reservoir",
            "max_quantile_samples": max_samples,
        },
    }
    if roi_meta is not None:
        metadata["roi"] = roi_meta
    metadata_path = make_unique_output_path(class_output_path.with_suffix(".json"))
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "class_raster": class_output_path,
        "value_raster": value_output_path,
        "world_file": class_world_file,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "class_counts": class_counts,
        "resolved_thresholds": resolved_thresholds,
        "valid_pixel_count": total_valid,
        "value_range": {"min": value_min, "max": value_max},
        "band_grid": {"width": width, "height": height},
    }
