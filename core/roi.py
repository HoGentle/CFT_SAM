"""感兴趣区域（ROI）工具。

流程：手工标点生成凸包，或点提示分割生成含空洞的真实轮廓 →
服务端把轮廓经纬度顶点换算到诊断网格像素坐标 → 按分块窗口做扫描线栅格化，
与有效掩膜求交。全图尺寸可达数十亿像素，因此掩膜必须逐块生成，不能整图驻留。
"""

import numpy as np


def convex_hull(points):
    """Andrew 单调链凸包（按像素坐标排序，与经纬度排序等价）。

    服务端对收到的顶点先重排为凸包顺序，保证即使客户端传入乱序
    或自交的顶点序列，栅格化结果仍是正确的凸包区域。
    """
    pts = np.asarray(points, dtype="float64")
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def build(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                cross = ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                         - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]))
                if cross <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lower = build(pts)
    upper = build(pts[::-1])
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype="float64")


def polygons_lonlat_to_pixel(polygons_lonlat, transform, crs=None):
    """经纬度转诊断网格；手工区域重排凸包，分割区域保留边界和空洞。"""
    from rasterio.warp import transform as project

    inverse = ~transform
    polygons_px = []

    def convert(points):
        pts = np.asarray(points, dtype="float64")
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            raise ValueError("多边形顶点必须是 [lon, lat] 数组")
        xs, ys = pts[:, 0], pts[:, 1]
        if crs is not None:
            xs, ys = project("EPSG:4326", crs, xs.tolist(), ys.tolist())
            xs, ys = np.asarray(xs), np.asarray(ys)
        cols, rows = inverse * (xs, ys)
        return np.stack([cols, rows], axis=1)

    for region in polygons_lonlat:
        if isinstance(region, dict):
            poly = convert(region["hull"])
            if region.get("source") == "sam":
                polygons_px.append({"hull": poly, "holes": [convert(h) for h in region.get("holes", [])]})
            else:
                polygons_px.append(convex_hull(poly))
        else:
            polygons_px.append(convex_hull(convert(region)))
    return polygons_px


def rasterize_polygons_in_window(polygons_px, window):
    """在窗口内栅格化多边形（偶奇规则，按像元中心判定），返回 bool 掩膜。

    polygons_px: 全局像素坐标的顶点数组，或包含 hull、holes 的区域字典。
    """
    height, width = int(window.height), int(window.width)
    mask = np.zeros((height, width), dtype=bool)
    row_off = int(window.row_off)
    col_off = int(window.col_off)

    for poly in polygons_px:
        if isinstance(poly, dict):
            # 自动轮廓可能含大量顶点，使用原生栅格化避免逐行遍历所有边。
            from affine import Affine
            from rasterio.features import rasterize
            geometry = {"type": "Polygon", "coordinates": [
                np.asarray(ring).tolist() for ring in [poly["hull"], *poly.get("holes", [])]
            ]}
            mask |= rasterize([(geometry, 1)], out_shape=(height, width),
                              transform=Affine.translation(col_off, row_off), dtype="uint8").astype(bool)
            continue
        x = poly[:, 0]
        y = poly[:, 1]
        n = len(poly)
        if n < 3:
            continue

        r_start = max(row_off, int(np.floor(y.min())))
        r_end = min(row_off + height, int(np.ceil(y.max())) + 1)
        c_lo = float(col_off)
        c_hi = float(col_off + width)

        for r in range(r_start, r_end):
            yc = r + 0.5
            xs = []
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if (y1 <= yc < y2) or (y2 <= yc < y1):
                    xs.append(x1 + (yc - y1) * (x2 - x1) / (y2 - y1))
            if not xs:
                continue
            xs.sort()
            row_mask = mask[r - row_off]
            for i in range(0, len(xs) - 1, 2):
                # 像元中心 c+0.5 ∈ [xs[i], xs[i+1]) -> c ∈ [ceil(xs[i]-0.5), ceil(xs[i+1]-0.5)-1]
                c0 = max(int(np.ceil(xs[i] - 0.5)), c_lo)
                c1 = min(int(np.ceil(xs[i + 1] - 0.5)), c_hi)
                if c1 > c0:
                    row_mask[int(c0) - col_off:int(c1) - col_off] = True

    return mask
