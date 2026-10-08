"""用本地 SAM 权重和正负点提示生成输入影像上的区域轮廓。"""

import threading
import gc
import importlib.util
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.features import rasterize, shapes

# 项目内的可选依赖目录，避免改变用户其他项目的运行环境。
MODEL_DEPS = Path(__file__).resolve().parents[1] / "workspace" / "model_deps"
if MODEL_DEPS.is_dir() and str(MODEL_DEPS) not in sys.path:
    sys.path.insert(0, str(MODEL_DEPS))

MODEL_SPECS = {
    "sam1_vit_h": {"label": "SAM 1 · ViT-H", "filename": "sam_vit_h_4b8939.pth",
                   "family": "sam1", "module": "segment_anything", "backbone": "vit_h"},
    "hq_sam_vit_l": {"label": "HQ-SAM · ViT-L", "filename": "sam_hq_vit_l.pth",
                    "family": "hq", "module": "segment_anything_hq", "backbone": "vit_l"},
    "sam3": {"label": "SAM 3", "filename": "sam3.pt", "family": "sam3", "module": "sam3"},
    "sam3_1": {"label": "SAM 3.1 · 点提示", "filename": "sam3.1_multiplex.pt",
               "family": "sam3_1", "module": "sam3"},
}
DEFAULT_MODEL_ID = "sam1_vit_h"


def validate_prompts(points, width, height):
    """校验输入影像上的提示坐标；至少一个保留点。"""
    if not isinstance(points, list) or not 1 <= len(points) <= 128:
        raise ValueError("请提供 1 到 128 个提示点")
    coords, labels = [], []
    for point in points:
        try:
            x, y = float(point["x"]), float(point["y"])
            label = point["label"]
        except (KeyError, TypeError, ValueError):
            raise ValueError("提示点必须包含有效的坐标和保留／排除标签") from None
        if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
            raise ValueError("提示点必须位于预览图内")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("提示点标签只能是 1（保留）或 0（排除）")
        coords.append([x, y])
        labels.append(label)
    if 1 not in labels:
        raise ValueError("请至少添加一个保留点")
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def contains_point(ring, x, y):
    """射线法判断点是否在轮廓内。"""
    inside = False
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[i - 1]
        if (yi > y) != (yj > y) and x < xi + (y - yi) * (xj - xi) / (yj - yi):
            inside = not inside
    return inside


def mask_to_regions(mask, coords, labels, excluded=None):
    """保留含正提示点的连通区域，扣除已完成区域，保留凹边界和空洞。"""
    mask = np.asarray(mask, dtype=bool).copy()
    if excluded:
        geometries = []
        for region in excluded:
            rings = [region["hull"], *region.get("holes", [])]
            geometries.append(({"type": "Polygon", "coordinates": rings}, 1))
        mask &= ~rasterize(geometries, out_shape=mask.shape, transform=Affine.identity()).astype(bool)
    positives = [(int(x) + 0.5, int(y) + 0.5) for (x, y), label in zip(coords, labels) if label == 1]
    regions = []
    for geometry, _ in shapes(mask.astype(np.uint8), mask=mask, connectivity=4):
        rings = geometry["coordinates"]
        if not any(contains_point(rings[0], x, y) and
                   not any(contains_point(hole, x, y) for hole in rings[1:])
                   for x, y in positives):
            continue
        regions.append({"hull": rings[0][:-1], "holes": [r[:-1] for r in rings[1:]]})
    if not regions:
        raise ValueError("未识别到包含保留点的可用区域，请调整提示点")
    return regions


class SamRoiService:
    """延迟加载模型，缓存当前影像编码，串行保护有状态预测器。"""

    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        self.checkpoint_dir = self.checkpoint.parent if self.checkpoint.suffix in (".pth", ".pt") else self.checkpoint
        self._lock = threading.Lock()
        self._predictor = None
        self._image_key = None
        self._loaded_model_id = None
        self.device = None

    def list_models(self):
        """只列出目录中实际存在且有对应加载器的权重。"""
        result = []
        for model_id, spec in MODEL_SPECS.items():
            if not (self.checkpoint_dir / spec["filename"]).is_file():
                continue
            available = importlib.util.find_spec(spec["module"]) is not None
            result.append({"id": model_id, "label": spec["label"], "filename": spec["filename"],
                           "available": available,
                           "reason": "" if available else "缺少模型依赖，请安装对应依赖"})
        return result

    def _release(self):
        self._predictor = None
        self._image_key = None
        self._loaded_model_id = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self, model_id):
        if self._predictor is not None and self._loaded_model_id == model_id:
            return
        spec = MODEL_SPECS[model_id]
        checkpoint = self.checkpoint_dir / spec["filename"]
        if not checkpoint.is_file():
            raise ValueError(f"未找到分割权重：{spec['filename']}")
        self._release()
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if spec["family"] == "sam1":
                from segment_anything import SamPredictor, sam_model_registry
            elif spec["family"] == "hq":
                from segment_anything_hq import SamPredictor, sam_model_registry
            else:
                from core.sam3_roi import build_point_predictor
                self._predictor = build_point_predictor(checkpoint, spec["family"], self.device)
                self._loaded_model_id = model_id
                return
            model = sam_model_registry[spec["backbone"]](checkpoint=str(checkpoint))
            model.to(device=self.device)
            model.eval()
            self._predictor = SamPredictor(model)
            self._loaded_model_id = model_id
        except ImportError as exc:
            raise RuntimeError(f"{spec['label']} 缺少依赖（{exc.name}），请安装对应模型依赖") from exc

    def predict(self, image, image_key, points, excluded=None, model_id=DEFAULT_MODEL_ID, valid_mask=None,
                gap_kernel=None):
        if not isinstance(model_id, str) or model_id not in MODEL_SPECS:
            raise ValueError("所选模型无效，请从模型列表选择")
        coords, labels = validate_prompts(points, image.shape[1], image.shape[0])
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("正在处理另一组提示点，请稍后重试")
        try:
            started = time.perf_counter()
            self._load(model_id)
            import torch
            # 第三代模型采用官方推荐的混合精度，减少显存占用。
            mixed = (torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=MODEL_SPECS[model_id]["family"].startswith("sam3"))
                     if self.device == "cuda" else nullcontext())
            with torch.inference_mode(), mixed:
                if self._image_key != image_key:
                    self._image_key = None
                    self._predictor.set_image(image)
                    self._image_key = image_key
                masks, scores, _ = self._predictor.predict(
                    point_coords=coords, point_labels=labels, multimask_output=True,
                )
            masks = np.asarray(masks, dtype=bool)
            if valid_mask is not None:
                masks = masks & np.asarray(valid_mask, dtype=bool)[None, :, :]
            scores = np.asarray(scores).reshape(-1)
            # 优先使用同时满足所有正负提示的候选，再按模型质量分选择。
            ix, iy = coords[:, 0].astype(int), coords[:, 1].astype(int)
            negative_ok = np.all(~masks[:, iy[labels == 0], ix[labels == 0]], axis=1)
            if gap_kernel is not None:
                from core.sam_boundary import close_small_gaps
                for candidate in np.flatnonzero(negative_ok):
                    masks[candidate] = close_small_gaps(masks[candidate], gap_kernel, coords, labels,
                                                       valid_mask, excluded)
            eligible = np.all(masks[:, iy, ix] == labels.astype(bool)[None, :], axis=1) & negative_ok
            candidates = np.flatnonzero(eligible)
            if not len(candidates):
                raise ValueError("当前候选未能同时满足保留和排除点，请调整提示点")
            best = int(candidates[np.argmax(scores[candidates])])
            return {"regions": mask_to_regions(masks[best], coords, labels, excluded),
                    "score": float(scores[best]), "device": self.device,
                    "model_id": model_id, "model_label": MODEL_SPECS[model_id]["label"],
                    "elapsed_seconds": round(time.perf_counter() - started, 3)}
        finally:
            self._lock.release()
