"""降采样定位完整田块，再用有重叠的原图窗口修正边界。"""

from collections import deque

import numpy as np
from affine import Affine
from PIL import Image, ImageFilter
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from shapely.geometry import GeometryCollection, Point, Polygon, box, mapping, shape
from shapely.ops import unary_union

from core.preview import _out_shape
from core.sam_boundary import _binary_filter_axis, gap_kernel_for_width
from core.sam_postprocess import _ring_area_m2


def _parts(geometry):
    if geometry.is_empty:
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _parts(part)
    else:
        yield geometry


def _polygonal(geometry):
    """窗口相切会产生零面积线段，合并时只保留面，避免污染后续边界。"""
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    return unary_union([p for p in _parts(geometry) if p.geom_type == "Polygon"])


def _geometry(regions, origin=(0, 0), scale=(1, 1)):
    def ring(points):
        return np.asarray(points) * scale + origin

    polygons = [Polygon(ring(r["hull"]), [ring(h) for h in r.get("holes", [])]) for r in regions]
    return unary_union([p if p.is_valid else p.buffer(0) for p in polygons])


def _regions(geometry, origin=(0, 0), scale=(1, 1)):
    def ring(points):
        return ((np.asarray(points)[:-1] - origin) / scale).tolist()

    return [{"hull": ring(p.exterior.coords), "holes": [ring(h.coords) for h in p.interiors]}
            for p in _parts(geometry) if p.geom_type == "Polygon" and p.area > 0]


def _window_box(window):
    return box(window.col_off, window.row_off,
               window.col_off + window.width, window.row_off + window.height)


def _mask(geometry, window):
    clipped = geometry.intersection(_window_box(window))
    size = (int(window.height), int(window.width))
    if clipped.is_empty:
        return np.zeros(size, dtype=bool)
    return rasterize([(mapping(clipped), 1)], out_shape=size,
                     transform=Affine.translation(window.col_off, window.row_off),
                     dtype="uint8").astype(bool)


def _mask_geometry(mask, window):
    return unary_union([shape(g) for g, value in shapes(
        mask.astype(np.uint8), mask=mask, connectivity=4,
        transform=Affine.translation(window.col_off, window.row_off)) if value])


def _sample_points(mask, count=4):
    """从不同象限的可靠内部取点，避免只提示某一局部纹理。"""
    height, width = mask.shape
    points = []
    for y0, y1 in ((0, height // 2), (height // 2, height)):
        for x0, x1 in ((0, width // 2), (width // 2, width)):
            tile = mask[y0:y1, x0:x1]
            if not tile.any():
                continue
            ys, xs = np.nonzero(tile)
            distance = (xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2
            index = int(np.argmin(distance))
            points.append([float(xs[index] + x0) + 0.5, float(ys[index] + y0) + 0.5])
    return points[:count]


def _cell_window(cell, step, width, height):
    x, y = cell[0] * step, cell[1] * step
    return Window(x, y, min(step, width - x), min(step, height - y))


def _boundary_cells(boundary, step, width, height):
    """按边界长度采样，再覆盖邻格，避免扫描整个田块的内部。"""
    cells = set()
    band = boundary.buffer(step * 0.55)
    for line in _parts(boundary):
        if not hasattr(line, "interpolate") or not line.length:
            continue
        for distance in np.arange(0, line.length + step / 2, step / 2):
            point = line.interpolate(min(distance, line.length))
            cx, cy = int(point.x // step), int(point.y // step)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cell = cx + dx, cy + dy
                    if (0 <= cell[0] * step < width and 0 <= cell[1] * step < height
                            and cell not in cells
                            and _window_box(_cell_window(cell, step, width, height)).intersects(band)):
                        cells.add(cell)
    return cells


def _validate_source_prompts(sources, composite, coords, labels, original_count, excluded, read_valid):
    for index, ((x, y), label) in enumerate(zip(coords, labels)):
        if label != 1:
            continue
        pixel = Window(int(x), int(y), 1, 1)
        if not read_valid(sources, pixel, composite)[0, 0]:
            raise ValueError("保留点落在原始影像无效像元上，请调整提示位置")
        if excluded.covers(Point(int(x) + 0.5, int(y) + 0.5)):
            name = "划线提示" if index >= original_count else "保留点"
            raise ValueError(f"{name}落在已确认区域内，请调整提示位置")


def _gap_options(settings, src, window, output_shape):
    if not settings["ignore_small_boundaries"]:
        return {}
    transform = src.window_transform(window) * Affine.scale(
        window.width / output_shape[1], window.height / output_shape[0])
    return {"gap_kernel": gap_kernel_for_width(settings["max_gap_width_m"], transform,
                                               src.crs, output_shape)}


def _coarse(service, sources, composite, preview_id, coords, labels, extent, excluded,
            model_id, options, settings, read_window):
    from core.sam_window import _touches_crop, _window_for_points

    width, height = sources[0].width, sources[0].height
    limit = int(options.get("coarse_long_side", 1024))
    padding = int(options.get("padding", 256))
    size = max(int(options.get("max_window_size", 8192)),
               int(np.ceil(np.max(np.ptp(extent, axis=0)))) + 1 + 2 * padding)
    attempts = 0
    while True:
        window = _window_for_points(extent, width, height, size, padding,
                                    max(width, height, size))
        output_shape = _out_shape(window.height, window.width, limit)
        scale = np.array([window.width / output_shape[1], window.height / output_shape[0]])
        origin = np.array([window.col_off, window.row_off])
        local_coords = (coords - origin) / scale
        local_points = [{"x": float(x), "y": float(y), "label": int(label)}
                        for (x, y), label in zip(local_coords, labels)]
        pixels = {}
        for (x, y), label in zip(local_coords.astype(int), labels):
            key = int(x), int(y)
            if key in pixels and pixels[key] != label:
                raise ValueError("降采样后保留点与排除点过近，请拉开提示位置或提高 coarse_long_side")
            pixels[key] = label
        rgb, valid = read_window(sources, window, composite, out_shape=output_shape)
        ix, iy = local_coords.astype(int).T
        if not valid[iy[labels == 1], ix[labels == 1]].all():
            raise ValueError("降采样后保留点附近没有有效影像，请提高 coarse_long_side 或调整提示位置")
        result = service.predict(rgb, (preview_id, "coarse", *window.flatten(), *output_shape),
                                 local_points, _regions(excluded, origin, scale), model_id=model_id,
                                 valid_mask=valid, **_gap_options(settings, sources[0], window, output_shape))
        attempts += 1
        source_geometry = _geometry(result["regions"], origin, scale)
        source_regions = _regions(source_geometry, origin)
        if not _touches_crop(source_regions, window, width, height):
            return result, source_geometry, scale, attempts, window, output_shape
        size = max(int(window.width), int(window.height)) * 2


def _local_prompts(coords, labels, window, current_mask, exterior_mask, valid, uncertainty):
    origin = np.array([window.col_off, window.row_off])
    local = coords - origin
    inside = ((local >= 0).all(axis=1) & (local[:, 0] < window.width)
              & (local[:, 1] < window.height))
    prompts = [{"x": float(x), "y": float(y), "label": int(label)}
               for (x, y), label in zip(local[inside], labels[inside])]
    safe = _binary_filter_axis(current_mask, 2 * uncertainty + 1, 1, False)
    safe = _binary_filter_axis(safe, 2 * uncertainty + 1, 0, False) & valid
    # 狭长田块可能被腐蚀为空，此时从已有轮廓内部传递提示。
    if not safe.any():
        safe = current_mask & valid
    used = {(int(p["x"]), int(p["y"])) for p in prompts}
    for mask, label in ((safe, 1), (exterior_mask & valid, 0)):
        for x, y in _sample_points(mask):
            if len(prompts) >= 128:
                break
            if (int(x), int(y)) not in used:
                prompts.append({"x": x, "y": y, "label": label})
                used.add((int(x), int(y)))
    if not any(p["label"] == 1 for p in prompts):
        raise ValueError("边界窗口缺少可靠保留点，请在遗漏田块内部补充保留点")
    return prompts, int(inside.sum())


def _predict_fine(service, rgb, key, prompts, original_count, excluded, **kwargs):
    """辅助提示不一定准确，冲突时保留所有用户提示并只传递一个内部点重试。"""
    try:
        return service.predict(rgb, key, prompts, excluded, **kwargs), 0
    except ValueError as exc:
        if "候选未能同时满足保留和排除点" not in str(exc) or len(prompts) == original_count:
            raise
        retry = prompts[:original_count]
        if not any(p["label"] == 1 for p in retry):
            retry = retry + [next(p for p in prompts[original_count:] if p["label"] == 1)]
        return service.predict(rgb, key, retry, excluded, **kwargs), 1


def _keep_prompt_components(geometry, coords, labels):
    positives = [Point(int(x) + 0.5, int(y) + 0.5) for (x, y), label in zip(coords, labels) if label == 1]
    selected = unary_union([p for p in _parts(geometry) if p.geom_type == "Polygon"
                            and any(p.covers(point) for point in positives)])
    for (x, y), label in zip(coords, labels):
        if selected.covers(Point(int(x) + 0.5, int(y) + 0.5)) != bool(label):
            raise ValueError("合并后的边界未能满足所有保留／排除点，请补充提示后重试")
    return selected


def _restore_source_validity(geometry, sources, composite, size, read_valid, margin=0):
    """流式检查田块内部的原图无效像元，保留低分辨率中可能消失的小洞。"""
    width, height = sources[0].width, sources[0].height
    left, top, right, bottom = geometry.bounds
    coverage = geometry.buffer(margin) if margin else geometry
    forbidden = []
    for y in range(max(0, int(top) - margin), min(height, int(np.ceil(bottom)) + margin), size):
        for x in range(max(0, int(left) - margin), min(width, int(np.ceil(right)) + margin), size):
            window = Window(x, y, min(size, width - x), min(size, height - y))
            if not coverage.intersects(_window_box(window)):
                continue
            valid = read_valid(sources, window, composite)
            if not valid.all():
                invalid = _mask_geometry(~valid, window)
                if not invalid.is_empty:
                    forbidden.append(invalid)
    blocked = unary_union(forbidden)
    return geometry.difference(blocked), blocked


def _postprocess(geometry, forbidden, sources, coords, labels, settings, size):
    """按完整洞面积处理去洞；外轮廓平滑也采用原图分块，避免整田栅格化。"""
    if settings["fill_holes"]:
        polygons = []
        negative = [Point(int(x) + 0.5, int(y) + 0.5)
                    for (x, y), label in zip(coords, labels) if label == 0]
        for polygon in _parts(geometry):
            if polygon.geom_type != "Polygon":
                continue
            holes = []
            for ring in polygon.interiors:
                hole = Polygon(ring)
                protected = (hole.intersection(forbidden).area > 0
                             or any(hole.covers(point) for point in negative))
                remove = False
                if not protected:
                    area = _ring_area_m2(list(ring.coords), sources[0].transform, sources[0].crs)
                    limit = settings["max_hole_area_m2"]
                    remove = area < limit and not np.isclose(area, limit, rtol=1e-8, atol=1e-10)
                if not remove:
                    holes.append(ring.coords)
            polygons.append(Polygon(polygon.exterior.coords, holes))
        geometry = unary_union(polygons)
    if settings["smooth_boundary"]:
        width, height = sources[0].width, sources[0].height
        outer = unary_union([Polygon(p.exterior) for p in _parts(geometry) if p.geom_type == "Polygon"])
        holes = outer.difference(geometry)
        result = outer
        for cell in sorted(_boundary_cells(outer.boundary, size, width, height)):
            core = _cell_window(cell, size, width, height)
            x, y = max(0, core.col_off - 16), max(0, core.row_off - 16)
            window = Window(x, y, min(width, core.col_off + core.width + 16) - x,
                            min(height, core.row_off + core.height + 16) - y)
            mask = np.asarray(Image.fromarray(_mask(outer, window).astype(np.uint8) * 255)
                              .filter(ImageFilter.GaussianBlur(radius=3))) >= 128
            area = _window_box(core)
            result = result.difference(area).union(_mask_geometry(mask, window).intersection(area))
        smoothed = result.difference(holes.union(forbidden))
        for (x, y), label in zip(coords, labels):
            if smoothed.covers(Point(int(x) + 0.5, int(y) + 0.5)) != bool(label):
                patch = box(int(x) - 3, int(y) - 3, int(x) + 4, int(y) + 4)
                smoothed = smoothed.difference(patch).union(geometry.intersection(patch))
        geometry = smoothed
    return _keep_prompt_components(geometry, coords, labels)


def predict_multiscale(service, sources, composite, preview_id, coords, labels, extent,
                       original_count, excluded_regions, preview_scale, model_id, options,
                       settings, read_window, read_valid):
    size = int(options.get("window_size", 2048))
    overlap = float(options.get("overlap", 0.7))
    limit = int(options.get("coarse_long_side", 1024))
    max_tiles = int(options.get("max_refinement_windows", 512))
    agreement = float(options.get("min_overlap_agreement", 0.6))
    if (not np.isfinite(overlap) or not 0.3 <= overlap <= 0.85
            or not 64 <= limit <= 4096 or not 1 <= max_tiles <= 4096
            or not np.isfinite(agreement) or not 0.5 <= agreement <= 1):
        raise ValueError("粗到细识别配置无效，请检查降采样尺寸、重叠率和窗口数量上限")
    width, height = sources[0].width, sources[0].height
    image_area = box(0, 0, width, height)
    excluded = _geometry(excluded_regions, scale=preview_scale)
    _validate_source_prompts(sources, composite, coords, labels, original_count, excluded, read_valid)
    result, coarse, coarse_scale, attempts, coarse_window, output_shape = _coarse(
        service, sources, composite, preview_id, coords, labels, extent, excluded,
        model_id, options, settings, read_window)
    coarse = _polygonal(coarse.intersection(image_area).difference(excluded))
    current = _keep_prompt_components(coarse, coords, labels)
    # 粗轮廓只用于引导，允许局部扩展两个原图窗口；触及该范围时报错而非裁断。
    envelope = coarse.buffer(size * 2).intersection(image_area)
    uncertainty = max(3, int(np.ceil(max(coarse_scale) * 4)))
    coarse_exterior = image_area.difference(coarse.buffer(max(uncertainty, size * 0.35)))
    step = max(1, int(round(size * (1 - overlap))))
    cells = _boundary_cells(current.boundary, step, width, height)
    center = coords[labels == 1].mean(axis=0) / step
    queue = deque(sorted(cells, key=lambda c: (c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2))
    pending, seen = set(cells), set()
    observed = GeometryCollection()
    scores = []
    prompt_retries = 0
    coarse_score = result.get("score")

    while True:
        while queue:
            cell = queue.popleft()
            pending.discard(cell)
            if cell in seen:
                continue
            if len(seen) >= max_tiles:
                raise ValueError("边界细化达到窗口数量上限，未输出截断结果；请补充提示或提高 max_refinement_windows")
            core = _cell_window(cell, step, width, height)
            area = _window_box(core)
            cx, cy = core.col_off + core.width / 2, core.row_off + core.height / 2
            w, h = min(size, width), min(size, height)
            window = Window(int(np.clip(np.floor(cx - w / 2), 0, width - w)),
                            int(np.clip(np.floor(cy - h / 2), 0, height - h)), w, h)
            rgb, valid = read_window(sources, window, composite)
            local_mask = _mask(current, window)
            local_excluded = _regions(excluded.intersection(_window_box(window)),
                                      (window.col_off, window.row_off))
            prompts, user_count = _local_prompts(coords, labels, window, local_mask,
                                                 _mask(coarse_exterior, window), valid, uncertainty)
            prediction, retries = _predict_fine(
                service, rgb, (preview_id, "fine", *window.flatten()), prompts, user_count,
                local_excluded, model_id=model_id, valid_mask=valid,
                **_gap_options(settings, sources[0], window, valid.shape))
            prompt_retries += retries
            candidate = _geometry(prediction["regions"], (window.col_off, window.row_off))
            # 只比较已细化的重叠区，并避开边界附近的不确定带。
            window_area = _window_box(window)
            local_geometry = _polygonal(current.intersection(window_area.buffer(uncertainty + 1)))
            trusted = observed.intersection(window_area).difference(area)
            trusted = trusted.difference(local_geometry.boundary.buffer(uncertainty))
            foreground = trusted.intersection(local_geometry)
            mismatch = trusted.intersection(candidate.symmetric_difference(local_geometry)).area
            anchor = local_geometry.buffer(-uncertainty).intersection(window_area)
            if ((trusted.area > 16 and 1 - mismatch / trusted.area < agreement)
                    or (foreground.area > 16
                        and foreground.intersection(candidate).area / foreground.area < agreement)
                    or (anchor.area > 16
                        and anchor.intersection(candidate).area / anchor.area < agreement)):
                raise ValueError("重叠窗口识别结果不一致，请在目标田块补充保留点、在相邻田块添加排除点")
            new_piece = _polygonal(candidate.intersection(area).difference(excluded))
            if new_piece.difference(envelope).area > 1:
                raise ValueError("局部扩展已远离粗轮廓，请补充提示改善完整田块定位后重试")
            old_piece = current.intersection(area)
            change = new_piece.symmetric_difference(old_piece)
            current = _polygonal(current.difference(area).union(new_piece))
            observed = observed.union(area)
            seen.add(cell)
            if "score" in prediction:
                scores.append(float(prediction["score"]))
            # 修正跨过中心区域边缘时，向邻格平移继续探索；覆盖斜向和凹形边界。
            if not change.is_empty:
                frontier = change.buffer(1)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        neighbor = cell[0] + dx, cell[1] + dy
                        if (neighbor in seen or neighbor in pending
                                or not 0 <= neighbor[0] * step < width
                                or not 0 <= neighbor[1] * step < height):
                            continue
                        if frontier.intersects(_window_box(_cell_window(neighbor, step, width, height))):
                            queue.append(neighbor)
                            pending.add(neighbor)
        unresolved = current.boundary.difference(image_area.boundary.buffer(1)).difference(observed.buffer(0.5))
        if unresolved.is_empty:
            break
        additions = _boundary_cells(unresolved, step, width, height) - seen
        if not additions:
            raise ValueError("田块边界尚未闭合，请补充提示后重新识别")
        queue.extend(sorted(additions))
        pending.update(additions)

    current = _keep_prompt_components(current, coords, labels)
    current, invalid = _restore_source_validity(current, sources, composite, size, read_valid,
                                               margin=16 if settings["smooth_boundary"] else 0)
    current = _postprocess(current, invalid.union(excluded), sources, coords, labels, settings, size)
    result = dict(result)
    result["regions"] = _regions(current)
    if scores:
        result["score"] = min(scores)
    result["inference_source"] = "original_tiff_coarse_to_fine"
    result["inference_window"] = {"x": int(coarse_window.col_off), "y": int(coarse_window.row_off),
                                  "width": int(coarse_window.width), "height": int(coarse_window.height),
                                  "attempts": attempts + len(seen)}
    result["multiscale"] = {"coarse_attempts": attempts, "coarse_shape": list(output_shape),
                            "coarse_score": coarse_score, "refinement_windows": len(seen),
                            "prompt_retries": prompt_retries,
                            "window_size": size, "overlap": overlap, "step": step,
                            "boundary_complete": True}
    return result
