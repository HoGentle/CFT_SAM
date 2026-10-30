"""区域矢量合并，以及原始 TIFF 滑窗边界扫描和折线引导分割。"""

import heapq
import math
from contextlib import ExitStack

import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageDraw
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from rasterio.warp import transform as project
from scipy import ndimage
from shapely import shortest_line
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Polygon, box, shape
from shapely.ops import split, transform as map_geometry, unary_union

from core.sam_window import _check_grid, _read_window


def _parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry] if not geometry.is_empty and geometry.area > 0 else []
    return [part for child in getattr(geometry, "geoms", []) for part in _parts(child)]


def _geometry(region, width, height):
    if not isinstance(region, dict):
        raise ValueError("区域格式无效")
    parts = region.get("parts") or [region]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 1000:
        raise ValueError("区域独立部分数量无效")
    polygons = []
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("holes", []), list):
            raise ValueError("区域轮廓格式无效")
        rings = []
        for ring in [part.get("hull"), *part.get("holes", [])]:
            pts = np.asarray(ring, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2 or not 3 <= len(pts) <= 100000 or not np.isfinite(pts).all():
                raise ValueError("区域轮廓顶点无效")
            if (pts < 0).any() or (pts[:, 0] > width).any() or (pts[:, 1] > height).any():
                raise ValueError("区域轮廓超出预览范围")
            rings.append(pts)
        polygon = Polygon(rings[0], rings[1:])
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("区域边界自交或面积为零，请调整区域后重试")
        polygons.append(polygon)
    return unary_union(polygons)


def _encode(geometry, georef):
    def ring(coords):
        xy = np.asarray(coords[:-1], dtype=float)
        xy = np.clip(xy, [0, 0], [georef["preview_width"], georef["preview_height"]])
        sx = xy[:, 0] * georef["source_width"] / georef["preview_width"]
        sy = xy[:, 1] * georef["source_height"] / georef["preview_height"]
        a, b, c, d, e, f = georef["transform"]
        xs, ys = a * sx + b * sy + c, d * sx + e * sy + f
        if not georef.get("crs"):
            raise ValueError("影像缺少坐标系，无法编辑地理区域")
        lons, lats = project(georef["crs"], "EPSG:4326", xs.tolist(), ys.tolist())
        return [{"x": float(x), "y": float(y), "lon": float(lon), "lat": float(lat)}
                for (x, y), lon, lat in zip(xy, lons, lats)]
    parts = [{"hull": ring(list(p.exterior.coords)),
              "holes": [ring(list(h.coords)) for h in p.interiors]} for p in _parts(geometry)]
    if not parts:
        raise ValueError("未得到有效区域")
    result = {**parts[0], "source": "edited", "points": []}
    if len(parts) > 1:
        result["parts"] = parts
    return result


def _merge_connected(geometries, georef, max_distance, excluded):
    """按原始边缘的最近邻连接树填补间隙，成功结果只能有一个外轮廓。"""
    if isinstance(max_distance, bool):
        raise ValueError("最大合并距离须为非负数，或 -1（不限距离）")
    try:
        limit = float(max_distance)
    except (TypeError, ValueError):
        raise ValueError("最大合并距离须为非负数，或 -1（不限距离）") from None
    if not math.isfinite(limit) or (limit < 0 and limit != -1):
        raise ValueError("最大合并距离须为非负数，或 -1（不限距离）")
    original = unary_union(geometries)
    if original.geom_type == "Polygon":
        return original, []
    if not georef.get("crs"):
        raise ValueError("影像缺少坐标系，无法计算区域间的米制距离")
    sx = georef["source_width"] / georef["preview_width"]
    sy = georef["source_height"] / georef["preview_height"]
    raster = Affine(*georef["transform"])
    center = original.centroid
    cx, cy = raster * (center.x * sx, center.y * sy)
    lons, lats = project(georef["crs"], "EPSG:4326", [cx], [cy])
    metric = f"+proj=aeqd +lat_0={lats[0]} +lon_0={lons[0]} +datum=WGS84 +units=m +no_defs"

    def to_metric(x, y, z=None):
        px, py = raster * (np.asarray(x) * sx, np.asarray(y) * sy)
        return project(georef["crs"], metric, px, py)

    def to_preview(x, y, z=None):
        px, py = project(metric, georef["crs"], x, y)
        px, py = (~raster) * (np.asarray(px), np.asarray(py))
        return px / sx, py / sy

    components = _parts(map_geometry(to_metric, original))
    # 只以原始区域计算距离；新建连接带不能作为跳板绕过距离限制。
    nearest = {i: (components[0].distance(components[i]), 0) for i in range(1, len(components))}
    links = []
    while nearest:
        target = min(nearest, key=lambda i: (nearest[i][0], i))
        distance, source = nearest.pop(target)
        if limit != -1 and distance > limit + 1e-6:
            raise ValueError(f"距离过远无法合并：最近边缘距离 {distance:.3f} 米，超过设置的 {limit:g} 米；可调整最大合并距离，-1 表示不限距离")
        links.append((source, target, distance))
        for index in nearest:
            candidate = components[target].distance(components[index])
            if candidate < nearest[index][0]:
                nearest[index] = (candidate, target)

    additions = []
    for source, target, distance in links:
        a, b = components[source], components[target]
        pair = unary_union([a, b])
        # 正负缓冲形成局部闭运算，填平相对边缘之间的细缝。
        # 仅接受同时接触两个面的新增部分，避免改动原有凹边和洞。
        radius = max(0.01, distance * 0.51)
        closed = pair.buffer(radius).buffer(-radius)
        fill = [p for p in _parts(closed.difference(pair))
                if p.distance(a) < 1e-7 and p.distance(b) < 1e-7]
        joined = unary_union([a, b, *fill])
        if joined.geom_type != "Polygon":
            # 角点接近或点接触时，最近线的有限宽缓冲保证面连通。
            half_width = max(0.01, min(0.5, distance * 0.1))
            fill.append(shortest_line(a, b).buffer(half_width))
        additions.extend(fill)
    # 在预览坐标中融合原始面，保留原始顶点精度，并限制到影像范围。
    connector = map_geometry(to_preview, unary_union(additions)).buffer(1e-7).intersection(
        box(0, 0, georef["preview_width"], georef["preview_height"]))
    if excluded and connector.difference(original).intersection(unary_union(excluded)).area > 1e-8:
        raise ValueError("连接区域经过未选中的感兴趣区域，请一并选择该区域后再合并")
    merged = unary_union([original, connector])
    if merged.geom_type != "Polygon" or not merged.is_valid:
        raise ValueError("无法在影像范围内连接成一个有效闭合区域，原区域保持不变")
    return merged, [float(distance) for _, _, distance in links]


def _scan_edges(sources, composite, parent):
    """逐窗读取原生像素，汇聚到有上限的分析网格，保留窗口重叠区。"""
    src = sources[0]
    left, top, right, bottom = parent.bounds
    left, top = max(0, int(left) - 16), max(0, int(top) - 16)
    right, bottom = min(src.width, int(math.ceil(right)) + 16), min(src.height, int(math.ceil(bottom)) + 16)
    stride = max(1, int(math.ceil(max(right - left, bottom - top) / 1536)))
    rows, cols = int(math.ceil((bottom - top) / stride)), int(math.ceil((right - left) / stride))
    transform = Affine(stride, 0, left, 0, stride, top)
    inside = rasterize([(parent, 1)], out_shape=(rows, cols), transform=transform, dtype="uint8").astype(bool)
    strength, valid_grid = np.zeros((rows, cols), dtype="float32"), np.zeros((rows, cols), dtype=bool)
    tile_size = max(stride, 1024 // stride * stride)
    scanned = 0
    for y in range(top, bottom, tile_size):
        for x in range(left, right, tile_size):
            w, h = min(tile_size, right - x), min(tile_size, bottom - y)
            if not parent.intersects(box(x, y, x + w, y + h)):
                continue
            lx, ty = max(0, x - 8), max(0, y - 8)
            rx, by = min(src.width, x + w + 8), min(src.height, y + h + 8)
            rgb, valid = _read_window(sources, Window(lx, ty, rx - lx, by - ty), composite)
            blurred = ndimage.gaussian_filter(rgb.astype("float32"), sigma=(2, 2, 0))
            gx = np.diff(blurred, axis=1, append=blurred[:, -1:])
            gy = np.diff(blurred, axis=0, append=blurred[-1:])
            edge = np.sqrt(np.mean(gx * gx + gy * gy, axis=2))
            edge = edge[y - ty:y - ty + h, x - lx:x - lx + w]
            usable = valid[y - ty:y - ty + h, x - lx:x - lx + w]
            nr, nc = int(math.ceil(h / stride)), int(math.ceil(w / stride))
            padding = ((0, nr * stride - h), (0, nc * stride - w))
            edge = np.pad(edge, padding).reshape(nr, stride, nc, stride).max(axis=(1, 3))
            usable = np.pad(usable, padding).reshape(nr, stride, nc, stride).any(axis=(1, 3))
            r, c = (y - top) // stride, (x - left) // stride
            strength[r:r + nr, c:c + nc] = edge
            valid_grid[r:r + nr, c:c + nc] = usable
            scanned += 1
    return strength, inside, valid_grid, transform, scanned


def _automatic_parts(parent, strength, inside, valid, transform):
    interior = ndimage.binary_erosion(inside, iterations=2) & valid
    values = strength[interior]
    if not len(values):
        raise ValueError("区域内可用像素不足，无法寻找新边界")
    median = float(np.median(values))
    threshold = max(4.0, float(np.percentile(values, 88)), median + 3 * float(np.median(abs(values - median))))
    barriers = ndimage.binary_closing(strength >= threshold, iterations=2)
    barriers = ndimage.binary_dilation(barriers)
    seeds, _ = ndimage.label(inside & valid & ~barriers)
    counts = np.bincount(seeds.ravel())
    keep = np.flatnonzero(counts >= max(16, int(inside.sum() * 0.005)))
    keep = keep[keep != 0]
    if not 2 <= len(keep) <= 64:
        raise ValueError("未找到可可靠分割的连续边界，请使用折线引导或调整目标区域")
    remap = np.zeros(len(counts), dtype="int32")
    remap[keep] = np.arange(1, len(keep) + 1)
    markers = remap[seeds]
    nearest = ndimage.distance_transform_edt(markers == 0, return_distances=False, return_indices=True)
    partition = markers[tuple(nearest)]
    pieces = {}
    for geometry, label in shapes(partition, transform=transform):
        clipped = shape(geometry).intersection(parent)
        if not clipped.is_empty and clipped.area > 0:
            pieces.setdefault(int(label), []).append(clipped)
    return [unary_union(parts) for parts in pieces.values()]


def _edge_path(strength, start, end):
    """沿折线附近的高对比度边界搜索路径，不用折线直接强制切图。"""
    height, width = strength.shape
    radius = max(4, min(24, int(min(height, width) * 0.03)))
    a, b = np.rint(start).astype(int), np.rint(end).astype(int)
    a = np.clip(a, [0, 0], [width - 1, height - 1])
    b = np.clip(b, [0, 0], [width - 1, height - 1])
    left, top = np.maximum(np.minimum(a, b) - radius, 0)
    right, bottom = np.minimum(np.maximum(a, b) + radius + 1, [width, height])
    tile = strength[top:bottom, left:right]
    a, b = a - [left, top], b - [left, top]
    guide = Image.new("L", (tile.shape[1], tile.shape[0]))
    ImageDraw.Draw(guide).line([tuple(a), tuple(b)], fill=1, width=2 * radius + 1)
    corridor = np.asarray(guide, dtype=bool)
    distance = np.full(tile.shape, np.inf, dtype="float32")
    previous = np.full(tile.shape, -1, dtype="int32")
    visited = np.zeros(tile.shape, dtype=bool)
    cost = 0.15 + 1 / (1 + tile / 2)
    distance[a[1], a[0]] = 0
    queue = [(0.0, 0.0, int(a[0]), int(a[1]))]
    while queue:
        _, travelled, x, y = heapq.heappop(queue)
        if visited[y, x]:
            continue
        visited[y, x] = True
        if (x, y) == tuple(b):
            path = []
            while True:
                path.append([x + int(left) + 0.5, y + int(top) + 0.5])
                parent = int(previous[y, x])
                if parent < 0:
                    return path[::-1]
                y, x = divmod(parent, tile.shape[1])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < tile.shape[1] and 0 <= ny < tile.shape[0]) or not corridor[ny, nx] or visited[ny, nx]:
                continue
            next_distance = travelled + float(cost[ny, nx]) * math.hypot(dx, dy)
            if next_distance >= distance[ny, nx]:
                continue
            distance[ny, nx] = next_distance
            previous[ny, nx] = y * tile.shape[1] + x
            estimate = next_distance + 0.15 * math.hypot(nx - b[0], ny - b[1])
            heapq.heappush(queue, (estimate, next_distance, nx, ny))
    raise ValueError("折线附近未找到连续路径，请调整折线位置")


def _guided_parts(parent, strength, inside, transform, guide):
    points = (~transform) * (guide[:, 0], guide[:, 1])
    points = np.column_stack(points)
    points = np.clip(points, [0, 0], [strength.shape[1] - 1, strength.shape[0] - 1])
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 0.5):
        raise ValueError("折线点过近，请删除重复点后重试")

    def extend(point, direction):
        limits = np.array([strength.shape[1] - 1, strength.shape[0] - 1])
        options = [(limits[i] - point[i]) / direction[i] if direction[i] > 0 else -point[i] / direction[i]
                   for i in (0, 1) if abs(direction[i]) > 1e-9]
        return point + min(t for t in options if t >= 0) * direction

    points = np.vstack([extend(points[0], points[0] - points[1]), points,
                        extend(points[-1], points[-1] - points[-2])])
    path = []
    for start, end in zip(points, points[1:]):
        if np.linalg.norm(end - start) < 0.5:
            continue
        segment = _edge_path(strength, start, end)
        path.extend(segment[1:] if path else segment)
    pixel_path = np.asarray(path)
    indices = np.clip(pixel_path.astype(int), [0, 0], [strength.shape[1] - 1, strength.shape[0] - 1])
    inner = ndimage.binary_erosion(inside, iterations=2)[indices[:, 1], indices[:, 0]]
    samples = strength[indices[inner, 1], indices[inner, 0]]
    if len(samples) < 2 or np.count_nonzero(samples >= 4) < max(2, len(samples) * 0.15):
        raise ValueError("折线附近没有清晰的影像边界，请将折线移到实际分界线附近")
    xs, ys = transform * (pixel_path[:, 0], pixel_path[:, 1])
    source_path = np.column_stack([xs, ys])
    distance = 3 * math.hypot(parent.bounds[2] - parent.bounds[0], parent.bounds[3] - parent.bounds[1])
    before, after = guide[0] - guide[1], guide[-1] - guide[-2]
    source_path = np.vstack([source_path[0] + before / np.linalg.norm(before) * distance,
                             source_path, source_path[-1] + after / np.linalg.norm(after) * distance])
    return _parts(split(parent, LineString(source_path)))


def edit_regions(preview, regions, operation, guide=None, merge_max_distance_m=5, excluded_regions=None):
    g = preview["georef"]
    if not isinstance(regions, list) or not 1 <= len(regions) <= 1000:
        raise ValueError("请选择有效区域")
    geometries = [_geometry(r, g["preview_width"], g["preview_height"]) for r in regions]
    if operation == "merge":
        if len(geometries) < 2:
            raise ValueError("至少选择两个区域才能合并")
        if not isinstance(excluded_regions or [], list) or len(excluded_regions or []) > 1000:
            raise ValueError("未选中区域格式无效")
        excluded = [_geometry(r, g["preview_width"], g["preview_height"]) for r in excluded_regions or []]
        merged, distances = _merge_connected(geometries, g, merge_max_distance_m, excluded)
        return {"regions": [_encode(merged, g)], "merge_distances_m": distances}
    if operation != "split" or len(geometries) != 1:
        raise ValueError("分割时必须只选择一个区域")
    sx, sy = g["source_width"] / g["preview_width"], g["source_height"] / g["preview_height"]
    parent = affine_transform(geometries[0], [sx, 0, 0, sy, 0, 0])
    line = np.asarray(guide if guide is not None else [], dtype=float)
    if line.size and (line.ndim != 2 or line.shape[1] != 2 or not 2 <= len(line) <= 256 or not np.isfinite(line).all()):
        raise ValueError("折线至少需要两个有效点，最多 256 个点")
    if line.size and ((line < 0).any() or (line[:, 0] > g["preview_width"]).any() or (line[:, 1] > g["preview_height"]).any()):
        raise ValueError("折线必须位于预览图内")
    paths = preview.get("source_paths") or {}
    composite = preview.get("mode") == "composite"
    roles = ("B4", "B3", "B2") if composite else ("preview",)
    if any(role not in paths for role in roles):
        raise ValueError("缺少原始影像，请重新生成预览")
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(paths[role])) for role in roles]
        for src in sources:
            _check_grid(src, g)
        strength, inside, valid, transform, scanned = _scan_edges(sources, composite, parent)
    pieces = (_guided_parts(parent, strength, inside, transform, line * [sx, sy]) if line.size
              else _automatic_parts(parent, strength, inside, valid, transform))
    pieces = [p for p in pieces if not p.is_empty and p.area > 1e-9]
    if not 2 <= len(pieces) <= 64:
        raise ValueError("未得到两个或更多有效区域，原区域保持不变，请调整折线或目标区域")
    pieces.sort(key=lambda p: (-p.area, p.bounds))
    return {"regions": [_encode(affine_transform(p, [1 / sx, 0, 0, 1 / sy, 0, 0]), g) for p in pieces],
            "scan_windows": scanned}
