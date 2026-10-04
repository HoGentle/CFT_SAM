"""按大疆模板替换处方图（移植自原 dj2pr.py）。

将自生成处方图重采样到设备模板（如大疆智图导出的 fertilizer.tif）的
网格上，并保留模板的标签/色彩表等结构，输出 new_fertilizer 成品。
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import reproject

from core.diagnose import make_unique_output_path, write_world_file

METHOD_SUFFIX = {"nearest": "nearest", "average": "mean"}


def has_usable_transform(transform):
    return transform is not None and transform != Affine.identity()


def is_north_up(transform):
    return abs(transform.b) < 1e-12 and abs(transform.d) < 1e-12


def copy_template_structure(src_template, dst_output):
    dataset_tags = src_template.tags()
    if dataset_tags:
        dst_output.update_tags(**dataset_tags)

    for band_index in range(1, src_template.count + 1):
        band_tags = src_template.tags(band_index)
        if band_tags:
            dst_output.update_tags(band_index, **band_tags)
        description = src_template.descriptions[band_index - 1]
        if description:
            dst_output.set_band_description(band_index, description)
        try:
            colormap = src_template.colormap(band_index)
        except ValueError:
            colormap = None
        if colormap:
            dst_output.write_colormap(band_index, colormap)


def resample_with_reproject(my_src, template_src, method):
    """双 CRS 场景用 GDAL 重投影/重采样（分块流式，不整图加载源）。

    源处方图无效像元为 NaN 且未写 nodata 标签，这里显式声明
    src_nodata=NaN，average 重采样时 GDAL 会排除无效像元。
    """
    destination = np.full(
        (template_src.height, template_src.width), np.nan, dtype=np.float32
    )
    reproject(
        source=rasterio.band(my_src, 1),
        destination=destination,
        src_transform=my_src.transform,
        src_crs=my_src.crs,
        src_nodata=np.nan,
        dst_transform=template_src.transform,
        dst_crs=template_src.crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest if method == "nearest" else Resampling.average,
    )
    valid_mask = np.isfinite(destination)
    return destination, valid_mask, f"reproject_{method}"


def _template_chunk_rows(my_src, template_src, max_source_rows=256):
    """估算每个模板行分块对应的源行数，保证分块读取的源窗口内存可控。"""
    src_pixel_h = abs(my_src.transform.e) or 1.0
    dst_pixel_h = abs(template_src.transform.e) or 1.0
    rows_per_dst = max(1e-9, dst_pixel_h / src_pixel_h)
    chunk = max(1, int(max_source_rows / rows_per_dst))
    return min(chunk, template_src.height)


def _src_valid_mask(data, nodata):
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    return valid


def resample_with_grid_nearest(my_src, template_src):
    """无 CRS 时按仿射变换做最近邻查表。

    按模板行分块、源图按需开窗读取（原实现整图读入，
    处方图解码后可达数 GB，会导致内存耗尽）。

    替换掩膜与原实现一致：模板范围内全部替换（源图 NaN 视为
    无效值照样替换成 NaN，nodata 标签值则不替换）。
    """
    dst_h, dst_w = template_src.height, template_src.width
    src_inverse = ~my_src.transform
    dst_cols = np.arange(dst_w, dtype=np.int64)
    chunk_rows = _template_chunk_rows(my_src, template_src)

    destination = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    replacement = np.zeros((dst_h, dst_w), dtype=bool)
    nodata = my_src.nodata

    for row0 in range(0, dst_h, chunk_rows):
        row1 = min(row0 + chunk_rows, dst_h)
        dst_rows = np.arange(row0, row1, dtype=np.int64)
        xs, ys = template_src.transform * (dst_cols + 0.5, dst_rows[:, None] + 0.5)
        src_cols_f, src_rows_f = src_inverse * (xs, ys)
        src_cols = np.floor(src_cols_f).astype(np.int64)
        src_rows = np.floor(src_rows_f).astype(np.int64)

        inside = (
            (src_rows >= 0) & (src_rows < my_src.height)
            & (src_cols >= 0) & (src_cols < my_src.width)
        )
        if not np.any(inside):
            continue

        r_min = int(src_rows[inside].min())
        r_max = int(src_rows[inside].max())
        c_min = int(src_cols[inside].min())
        c_max = int(src_cols[inside].max())
        window = Window(c_min, r_min, c_max - c_min + 1, r_max - r_min + 1)
        src_block = my_src.read(1, window=window).astype(np.float32)

        values = src_block[src_rows[inside] - r_min, src_cols[inside] - c_min]

        # 与原实现一致：采样到 nodata 标签值的位置不替换（保留模板原值）；
        # 其余范围内位置全部替换，无效(NaN)样本照原样写成 NaN。
        replace_here = inside.copy()
        if nodata is not None:
            replace_here[inside] &= values != nodata

        values_2d = np.full(inside.shape, np.nan, dtype=np.float32)
        values_2d[inside] = values

        block_dest = destination[row0:row1]
        block_dest[replace_here] = values_2d[replace_here]
        replacement[row0:row1] |= replace_here

    if not np.any(replacement):
        raise ValueError("模板范围与待替换处方图没有重叠区域，无法完成替换。")
    return destination, replacement, "grid_nearest"


def resample_with_grid_average(my_src, template_src):
    """无 CRS 时的平均池化：每个模板像元取其覆盖范围内有效像元均值。

    北向上栅格前提下按模板行分块：一次读取块对应的源行窗口，
    用行方向累加 + 列方向前缀和求每个模板像元的均值（原实现整图
    读入并逐像元双重循环，大图既耗内存又极慢）。
    """
    if not is_north_up(my_src.transform) or not is_north_up(template_src.transform):
        raise ValueError("当前平均池化仅支持无旋转的北向上栅格。")

    dst_h, dst_w = template_src.height, template_src.width
    src_inverse = ~my_src.transform
    my_transform = my_src.transform
    template_transform = template_src.transform

    chunk_rows = _template_chunk_rows(my_src, template_src, max_source_rows=256)
    destination = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    replacement = np.zeros((dst_h, dst_w), dtype=bool)

    for row0 in range(0, dst_h, chunk_rows):
        row1 = min(row0 + chunk_rows, dst_h)

        # 该模板行块覆盖的源行/列范围（北向上：行界只依赖行，列界只依赖列）
        y_top = template_transform.f + row0 * template_transform.e
        y_bottom = template_transform.f + row1 * template_transform.e
        x_left = template_transform.c
        x_right = template_transform.c + template_transform.a * dst_w

        _, top_f = src_inverse * (x_left, max(y_top, y_bottom))
        _, bottom_f = src_inverse * (x_left, min(y_top, y_bottom))
        left_f, _ = src_inverse * (min(x_left, x_right), y_top)
        right_f, _ = src_inverse * (max(x_left, x_right), y_top)

        row_start = max(0, int(np.floor(min(top_f, bottom_f))))
        row_end = min(my_src.height, int(np.ceil(max(top_f, bottom_f))))
        col_start = max(0, int(np.floor(min(left_f, right_f))))
        col_end = min(my_src.width, int(np.ceil(max(left_f, right_f))))
        if row_start >= row_end or col_start >= col_end:
            continue

        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
        src_block = my_src.read(1, window=window).astype(np.float32)
        src_valid = _src_valid_mask(src_block, my_src.nodata)
        filled = np.where(src_valid, src_block, 0.0).astype(np.float64)
        valid_f = src_valid.astype(np.float64)

        # 行方向累加：RS[i, :] = 前 i 行的和（含全部块内行）
        row_cumsum = np.cumsum(filled, axis=0)
        cnt_cumsum = np.cumsum(valid_f, axis=0)

        dst_cols = np.arange(dst_w, dtype=np.int64)
        x_lo = template_transform.c + dst_cols * template_transform.a
        x_hi = x_lo + template_transform.a
        col_f_lo, _ = src_inverse * (np.minimum(x_lo, x_hi), y_top)
        col_f_hi, _ = src_inverse * (np.maximum(x_lo, x_hi), y_top)
        col_starts = np.clip(np.floor(col_f_lo).astype(np.int64), 0, my_src.width)
        col_ends = np.clip(np.ceil(col_f_hi).astype(np.int64), 0, my_src.width)

        for row in range(row0, row1):
            y_a = template_transform.f + row * template_transform.e
            y_b = y_a + template_transform.e
            _, row_f_a = src_inverse * (x_left, y_a)
            _, row_f_b = src_inverse * (x_left, y_b)
            rs = max(row_start, int(np.floor(min(row_f_a, row_f_b))))
            re = min(row_end, int(np.ceil(max(row_f_a, row_f_b))))
            if rs >= re:
                continue
            rs_local = rs - row_start
            re_local = re - row_start

            row_sum_top = row_cumsum[rs_local - 1] if rs_local > 0 else 0.0
            row_sum = row_cumsum[re_local - 1] - row_sum_top
            row_cnt_top = cnt_cumsum[rs_local - 1] if rs_local > 0 else 0.0
            row_cnt = cnt_cumsum[re_local - 1] - row_cnt_top

            cs = np.clip(col_starts - col_start, 0, row_sum.size)
            ce = np.clip(col_ends - col_start, 0, row_sum.size)
            zero = np.zeros(1, dtype=np.float64)
            sums = np.concatenate([zero, np.cumsum(row_sum)])
            cnts = np.concatenate([zero, np.cumsum(row_cnt)])
            window_sum = sums[ce] - sums[cs]
            window_cnt = cnts[ce] - cnts[cs]

            out_row = np.full(dst_w, np.nan, dtype=np.float32)
            valid_block = window_cnt > 0
            if np.any(valid_block):
                out_row[valid_block] = (window_sum[valid_block] / window_cnt[valid_block]).astype(np.float32)
                replacement[row, :] |= valid_block
            destination[row, :] = out_row

    if not np.any(replacement):
        raise ValueError("模板范围与待替换处方图没有有效重叠区域，无法完成替换。")
    return destination, replacement, "grid_average_pooling"


def resample_to_template_grid(my_src, template_src, method):
    if my_src.crs and template_src.crs:
        return resample_with_reproject(my_src, template_src, method)
    if method == "nearest":
        return resample_with_grid_nearest(my_src, template_src)
    return resample_with_grid_average(my_src, template_src)


def convert_by_template(dj_path, my_path, new_fertilizer_path, method):
    """把 my_path 处方图重采样到 dj_path 模板网格并保留模板结构。"""
    if method not in METHOD_SUFFIX:
        raise ValueError(f"不支持的重采样方法: {method!r}（仅支持 nearest/average）")
    new_fertilizer_path = Path(new_fertilizer_path)
    new_fertilizer_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dj_path) as template_src, rasterio.open(my_path) as my_src:
        template_profile = template_src.profile.copy()
        template_transform = template_src.transform
        my_transform = my_src.transform

        resampled_data, replacement_mask, resampling_mode = resample_to_template_grid(
            my_src, template_src, method
        )
        template_bands = template_src.read()
        output_bands = template_bands.copy()
        output_band = output_bands[0]
        output_band[replacement_mask] = resampled_data[replacement_mask].astype(
            output_band.dtype, copy=False,
        )

        new_fertilizer_path = make_unique_output_path(new_fertilizer_path)
        with rasterio.open(new_fertilizer_path, "w", **template_profile) as dst:
            dst.write(output_bands)
            copy_template_structure(template_src, dst)

        world_file = write_world_file(new_fertilizer_path, template_transform)

    return {
        "dj_path": str(dj_path),
        "my_path": str(my_path),
        "new_fertilizer_path": str(new_fertilizer_path),
        "world_file": str(world_file),
        "template_size": [int(template_profile["width"]), int(template_profile["height"])],
        "template_dtype": str(template_profile["dtype"]),
        "resampling": resampling_mode,
        "method_suffix": METHOD_SUFFIX[method],
        "replaced_pixel_count": int(np.count_nonzero(replacement_mask)),
    }
