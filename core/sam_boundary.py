"""按地面宽度闭合细窄分隔，在连通区域筛选前处理模型掩膜。"""

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import transform as project

from core.prescription import utm_crs_for


def gap_kernel_for_width(width_m, transform, crs, shape):
    """在窗口中心量测原图像元地面尺寸，将米制阈值换算为矩形结构元素。"""
    if crs is None:
        raise ValueError("原始影像缺少坐标系，无法按米设置细小分隔宽度")
    height, width = shape
    x, y = width / 2, height / 2
    xs, ys = transform * (np.array([x, x + 1, x]), np.array([y, y, y + 1]))
    lons, lats = project(crs, "EPSG:4326", [xs[0]], [ys[0]])
    utm = utm_crs_for(lons[0], lats[0])
    mx, my = project(crs, utm, xs.tolist(), ys.tolist())
    sizes = np.hypot(np.asarray(mx)[1:] - mx[0], np.asarray(my)[1:] - my[0])
    if not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("无法计算原图像元地面宽度，请检查影像地理参考")
    # 闭运算跨越的最大轴向缝隙为结构元素长度减一；小于一个像元时不扩张。
    kx = min(2 * width + 1, int(np.floor(width_m / sizes[0] + 1e-8)) + 1)
    ky = min(2 * height + 1, int(np.floor(width_m / sizes[1] + 1e-8)) + 1)
    return kx, ky


def _binary_filter_axis(mask, size, axis, dilate):
    """用一维前缀和计算矩形膨胀／腐蚀，不增加图像处理依赖。"""
    if size <= 1:
        return mask
    before, after = (size - 1) // 2, size // 2
    if not dilate:
        before, after = after, before  # 偶数结构元素反转锚点，防止结果平移。
    padding = [(0, 0), (0, 0)]
    padding[axis] = (before, after)
    padded = np.pad(mask, padding, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.uint32)
    padding[axis] = (1, 0)
    cumulative = np.pad(cumulative, padding, mode="constant")
    low, high = [slice(None), slice(None)], [slice(None), slice(None)]
    low[axis] = slice(0, mask.shape[axis])
    high[axis] = slice(size, size + mask.shape[axis])
    sums = cumulative[tuple(high)] - cumulative[tuple(low)]
    return sums > 0 if dilate else sums == size


def close_small_gaps(mask, kernel, coords, labels, valid=None, excluded=None):
    """闭合细窄缝隙，保留原有前景、无效像元、已有区域和正负提示约束。"""
    kx, ky = kernel
    original = np.asarray(mask, dtype=bool)
    closed = _binary_filter_axis(original, kx, 1, True)
    closed = _binary_filter_axis(closed, ky, 0, True)
    closed = _binary_filter_axis(closed, kx, 1, False)
    closed = _binary_filter_axis(closed, ky, 0, False)
    closed = closed | original
    if valid is not None:
        closed &= np.asarray(valid, dtype=bool)
    if excluded:
        geometries = [({"type": "Polygon", "coordinates": [r["hull"], *r.get("holes", [])]}, 1)
                      for r in excluded]
        closed &= ~rasterize(geometries, out_shape=closed.shape, transform=Affine.identity()).astype(bool)
    # 排除点不会因细缝合并变为前景；候选已满足的保留点也必须保留。
    ix, iy = np.asarray(coords).astype(int).T
    closed[iy[labels == 0], ix[labels == 0]] = False
    return closed
