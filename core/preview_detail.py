"""按视野读取原图，限制单次输出、并发读取及全局色阶样本大小。"""

from contextlib import ExitStack
from functools import lru_cache
from io import BytesIO
import math
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window


MAX_SIDE = 2048
MAX_PIXELS = 2_000_000
READ_SLOT = BoundedSemaphore(1)


def validate_view(georef, bounds, size):
    """预览坐标保持不变，只调整读取窗口及其输出采样密度。"""
    x, y, width, height = map(float, bounds)
    out_w, out_h = map(int, size)
    if not all(math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError("视野坐标必须为有限数值")
    if (x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > georef["preview_width"] + 1e-6
            or y + height > georef["preview_height"] + 1e-6):
        raise ValueError("视野超出影像范围")
    if not (0 < out_w <= MAX_SIDE and 0 < out_h <= MAX_SIDE and out_w * out_h <= MAX_PIXELS):
        raise ValueError("视野输出超过像素限制")
    sx = georef["source_width"] / georef["preview_width"]
    sy = georef["source_height"] / georef["preview_height"]
    return Window(x * sx, y * sy, width * sx, height * sy), out_w, out_h


@lru_cache(maxsize=16)
def _stretch_limits(signature, indexes):
    """只缓存几个数值；各视野共用全图小样本色阶，避免拖动时亮度跳变。"""
    path = signature[0]
    with rasterio.open(path) as src:
        scale = min(1, 256 / max(src.width, src.height))
        shape = (len(indexes), max(1, round(src.height * scale)), max(1, round(src.width * scale)))
        sample = src.read(indexes, out_shape=shape, masked=True,
                          out_dtype="float32", resampling=Resampling.nearest)
        limits = []
        for index, band in zip(indexes, sample):
            if src.dtypes[index - 1] == "uint8":
                limits.append((0.0, 255.0))
                continue
            valid = band.compressed()
            valid = valid[np.isfinite(valid)]
            low, high = np.percentile(valid, (2, 98)) if valid.size else (0, 1)
            limits.append((float(low), float(high) if high > low else float(low + 1)))
        return tuple(limits)


def render_detail(preview, bounds, size):
    """优先利用已有概览层；仅返回视野图片，不保存全幅数组或无界瓦片缓存。"""
    window, out_w, out_h = validate_view(preview["georef"], bounds, size)
    paths = preview["source_paths"]
    # GDAL 缓存上限为 32 MiB；句柄在每次请求结束时关闭。
    with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024, GDAL_NUM_THREADS="1"), ExitStack() as stack:
        if "preview" in paths:
            src = stack.enter_context(rasterio.open(paths["preview"]))
            sources = [(src, index) for index in ([1, 2, 3] if src.count >= 3 else [1, 1, 1])]
        else:
            sources = [(stack.enter_context(rasterio.open(paths[role])), 1) for role in ("B4", "B3", "B2")]
        rgba = np.empty((out_h, out_w, 4), dtype=np.uint8)
        for channel, (src, index) in enumerate(sources):
            path = Path(src.name)
            stat = path.stat()
            limits = preview["georef"].get("display_limits")
            if limits:
                low, high = limits[channel]
            elif src.dtypes[index - 1] == "uint8":
                low, high = 0, 255
            else:
                low, high = _stretch_limits((str(path), stat.st_mtime_ns, stat.st_size), (index,))[0]
            data = src.read(index, window=window, out_shape=(out_h, out_w),
                            out_dtype="float32", resampling=Resampling.nearest)
            data -= low
            data *= 255 / (high - low)
            np.nan_to_num(data, copy=False, nan=0, posinf=255, neginf=0)
            np.clip(data, 0, 255, out=data)
            rgba[..., channel] = data
        rgba[..., 3] = sources[0][0].dataset_mask(
            window=window, out_shape=(out_h, out_w), resampling=Resampling.nearest)
        output = BytesIO()
        Image.fromarray(rgba).save(output, format="PNG")
        output.seek(0)
        return output
