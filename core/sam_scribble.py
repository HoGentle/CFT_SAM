"""将鼠标划线采样为模型保留点提示，不直接修改分割掩膜。"""

import numpy as np


def validate_scribbles(scribbles, width, height):
    if not isinstance(scribbles, list) or len(scribbles) > 63:
        raise ValueError("划线提示最多 63 条，请撤销部分线提示")
    result, total = [], 0
    for stroke in scribbles:
        if not isinstance(stroke, dict):
            raise ValueError("划线提示格式无效")
        try:
            path = np.asarray(stroke.get("points"), dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("划线提示坐标无效") from None
        if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 2
                or not np.isfinite(path).all()):
            raise ValueError("划线提示至少需要两个有效路径点")
        total += len(path)
        if total > 20000:
            raise ValueError("划线路径过长，最多 20000 个路径点")
        if ((path < 0).any() or (path[:, 0] > width).any()
                or (path[:, 1] > height).any()):
            raise ValueError("划线提示必须位于预览图内")
        if not np.any(np.diff(path, axis=0)):
            raise ValueError("请拖动鼠标画线，单击不能作为线提示")
        result.append({"points": path.tolist()})
    return result


def sample_scribbles(scribbles, scale, budget):
    """按原图弧长均匀取点，保留各条线首尾；返回采样点和全部路径顶点。"""
    if not scribbles:
        return np.empty((0, 2)), np.empty((0, 2))
    if budget < 2 * len(scribbles):
        raise ValueError("提示点数量不足以容纳所有划线，请撤销部分线或离散点（合计最多 128 点）")
    paths = [np.asarray(stroke["points"]) * scale for stroke in scribbles]
    arcs = [np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            for path in paths]
    desired = np.array([min(32, max(2, int(np.ceil(arc[-1] / 32)) + 1)) for arc in arcs])
    counts = np.full(len(paths), 2)
    # 优先给采样间距较大的线分配剩余预算，原有正负提示点始终保留。
    while counts.sum() < budget and np.any(counts < desired):
        index = max((i for i in range(len(paths)) if counts[i] < desired[i]),
                    key=lambda i: arcs[i][-1] / (counts[i] - 1))
        counts[index] += 1
    sampled = []
    for path, arc, count in zip(paths, arcs, counts):
        keep = np.r_[True, np.diff(arc) > 0]
        distances = np.linspace(0, arc[-1], count)
        sampled.append(np.column_stack([np.interp(distances, arc[keep], path[keep, axis])
                                        for axis in (0, 1)]))
    return np.concatenate(sampled), np.concatenate(paths)
