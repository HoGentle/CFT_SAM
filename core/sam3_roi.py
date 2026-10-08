"""将官方 SAM 3／3.1 的图像点提示分支适配为统一预测接口。

仅加载视觉骨干和点提示分支；不调用文字检测或视频传播。
SAM 3.1 使用其独立的交互分支，不能按 SAM 3 的参数结构加载。
"""

def load_point_weights(model, checkpoint, family):
    """映射本地合并权重，严格核对所用模型全部参数，避免随机参数混入。"""
    import torch
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    state = state.get("model", state)
    tracker_prefix = "tracker.model." if family == "sam3_1" else "tracker."
    mapped = {}
    for key, value in state.items():
        if key.startswith(tracker_prefix):
            mapped[key[len(tracker_prefix):]] = value
        elif key.startswith("detector.backbone.vision_backbone."):
            mapped[key.replace("detector.backbone.", "backbone.", 1)] = value
    expected = model.state_dict()
    missing = [key for key in expected if key not in mapped]
    if missing:
        raise RuntimeError(f"权重与点提示模型结构不匹配，缺少 {len(missing)} 项参数：{missing[:3]}")
    model.load_state_dict({key: mapped[key] for key in expected}, strict=True)


def build_point_predictor(checkpoint, family, device):
    import torch
    # 官方模块导入会修改全局计算设置，恢复它们以保持模型对比条件一致。
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        from sam3.model_builder import build_tracker, build_sam3_multiplex_video_model
        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32

    if family == "sam3":
        model = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
        # 官方跟踪器在构造时永久进入混合精度上下文；网页请求必须自行管理。
        model.bf16_context.__exit__(None, None, None)
        load_point_weights(model, checkpoint, family)
        model.to(device=device).eval()
        # 不填充小空洞，保持田埂、沟渠等排除区域。
        return SAM3InteractiveImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)

    model = build_sam3_multiplex_video_model(
        checkpoint_path=None, load_from_HF=False, device="cpu", compile=False,
        use_fa3=False, use_rope_real=True,
    )
    load_point_weights(model, checkpoint, family)
    model.to(device=device).eval()
    return Sam31PointPredictor(model, device)


class Sam31PointPredictor:
    """使用 SAM 3.1 官方视觉编码器、交互提示编码器和交互掩膜解码器。"""

    def __init__(self, model, device):
        from sam3.model.utils.sam1_utils import SAM2Transforms
        self.model = model
        self.device = device
        self.transform = SAM2Transforms(model.image_size, mask_threshold=0,
                                        max_hole_area=0, max_sprinkle_area=0)
        self.features = None
        self.image_shape = None

    def set_image(self, image):
        self.features = None
        self.image_shape = image.shape[:2]
        batch = self.transform(image)[None].to(self.device)
        backbone = self.model.forward_image(batch, need_interactive_out=True)
        self.features = self.model._prepare_backbone_features(backbone)["interactive"]

    def predict(self, point_coords, point_labels, multimask_output=True):
        import torch
        import torch.nn.functional as functional
        coords = self.transform.transform_coords(
            torch.as_tensor(point_coords, device=self.device, dtype=torch.float32),
            normalize=True, orig_hw=self.image_shape,
        )[None]
        labels = torch.as_tensor(point_labels, device=self.device, dtype=torch.int32)[None]
        sparse, dense = self.model.interactive_sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        features = self.features["vision_feats"]
        sizes = self.features["feat_sizes"]
        high_res = [feat.permute(1, 2, 0).reshape(1, -1, *size)
                    for feat, size in zip(features[:-1], sizes[:-1])]
        embedding = self.model._get_interactive_pix_mem(features, sizes)
        logits, scores, _, object_score = self.model.interactive_sam_mask_decoder(
            image_embeddings=embedding,
            image_pe=self.model.interactive_sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=multimask_output, repeat_image=True, high_res_features=high_res,
        )
        logits = torch.where(object_score[:, :, None, None] > 0, logits, -1024.0)
        masks = functional.interpolate(logits.float(), size=self.image_shape, mode="bilinear", align_corners=False) > 0
        return (masks[0].detach().cpu().numpy(), scores[0].float().detach().cpu().numpy(),
                logits[0].float().detach().cpu().numpy())
