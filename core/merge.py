"""分组结果栅格合并（分块流式）。

各分组的诊断分级图/诊断值图/处方图都在同一诊断网格上生成，且分组区域
互斥（区域编辑器保证区域不重叠、分组把区域划分开），因此可以按块读取
合并为单一输出：

* 浮点栅格（诊断值图/处方图）以 NaN 为无效，只接收有限值；
* 整型栅格（诊断分级图）以 nodata 值（255）为无效，只接收非 nodata 值；
* 分组区域互斥保证合并时无有效值冲突。
"""

import contextlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from core.diagnose import build_compatible_gtiff_profile, make_unique_output_path, write_world_file


def merge_group_rasters(group_paths, output_path, progress=None):
    """把同网格的多个单波段栅格合并为一个（网格/_dtype 必须一致）。

    返回 {"output_path", "world_file"}。
    """
    group_paths = [Path(p) for p in group_paths]
    if not group_paths:
        raise ValueError("没有可合并的分组栅格。")
    output_path = make_unique_output_path(Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(group_paths[0]) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        dtype = profile["dtype"]
        nodata = ref.nodata
        height, width = int(ref.height), int(ref.width)
        for p in group_paths[1:]:
            with rasterio.open(p) as g:
                if (int(g.width), int(g.height)) != (width, height):
                    raise ValueError(f"分组栅格尺寸不一致，无法合并: {p.name}")
                if g.transform != transform:
                    raise ValueError(f"分组栅格网格不一致，无法合并: {p.name}")
                if g.dtypes[0] != dtype:
                    raise ValueError(f"分组栅格数据类型不一致，无法合并: {p.name}")

    is_float = np.issubdtype(np.dtype(dtype), np.floating)
    out_profile = build_compatible_gtiff_profile(
        src_profile=profile,
        dtype=dtype,
        count=1,
        include_crs=ref.crs is not None,
        nodata_value=None if is_float else nodata,
    )

    block = 1024
    total = ((height + block - 1) // block) * ((width + block - 1) // block)
    done = 0
    with contextlib.ExitStack() as stack:
        srcs = [stack.enter_context(rasterio.open(p)) for p in group_paths]
        dst = stack.enter_context(rasterio.open(output_path, "w", **out_profile))
        for row0 in range(0, height, block):
            row_h = min(block, height - row0)
            for col0 in range(0, width, block):
                col_w = min(block, width - col0)
                window = Window(col0, row0, col_w, row_h)
                if is_float:
                    merged = np.full((row_h, col_w), np.nan, dtype=dtype)
                    for src in srcs:
                        values = src.read(1, window=window)
                        mask = np.isfinite(values)
                        merged[mask] = values[mask]
                else:
                    fill = nodata if nodata is not None else 255
                    merged = np.full((row_h, col_w), fill, dtype=dtype)
                    for src in srcs:
                        values = src.read(1, window=window)
                        mask = values != fill
                        merged[mask] = values[mask]
                dst.write(merged, window=window, indexes=1)
                done += 1
                if progress and done % 10 == 0:
                    progress(done / total, f"合并分组栅格 {done}/{total} 块")

    world_file = write_world_file(output_path, transform)
    return {"output_path": output_path, "world_file": world_file}
