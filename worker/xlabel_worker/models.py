"""内置检测模型（torchvision，BSD 许可）"""

from __future__ import annotations

from pathlib import Path

import torch

from .protocol import log

# 名称和说明给界面展示
MODELS = {
    "frcnn_mobilenet": {
        "name": "Faster R-CNN MobileNetV3",
        "desc": "速度快、显存占用小，适合入门和没有独立显卡的电脑",
        "builder": "fasterrcnn_mobilenet_v3_large_fpn",
    },
    "frcnn_resnet50": {
        "name": "Faster R-CNN ResNet50 v2",
        "desc": "精度更高，训练和推理更慢，建议有 6GB 以上显存",
        "builder": "fasterrcnn_resnet50_fpn_v2",
    },
}


def build(key: str, num_classes: int, img_size: int, pretrained: bool = True) -> torch.nn.Module:
    """num_classes 不含背景。预训练权重下载失败时退回只用骨干网络权重，再不行就随机初始化。"""
    from torchvision.models import detection as D
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if key not in MODELS:
        raise ValueError(f"不支持的模型：{key}")
    builder = getattr(D, MODELS[key]["builder"])
    size = dict(min_size=img_size, max_size=int(img_size * 1.667))
    model = None
    if pretrained:
        for kw, what in (
            (dict(weights="DEFAULT"), "COCO 预训练权重"),
            (dict(weights=None, weights_backbone="DEFAULT"), "骨干网络预训练权重"),
        ):
            try:
                model = builder(**kw, **size)
                log(f"已加载 {what}")
                break
            except Exception as e:  # 多半是网络问题
                log(f"加载 {what} 失败：{e}")
    if model is None:
        if pretrained:
            log("没有可用的预训练权重，从头训练，效果会差很多")
        model = builder(weights=None, weights_backbone=None, **size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def save(path: Path, model: torch.nn.Module, key: str, classes: list[str], img_size: int, extra: dict | None = None):
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "xlabel-det-1",
            "model": key,
            "classes": classes,
            "img_size": img_size,
            "state_dict": model.state_dict(),
            **(extra or {}),
        },
        tmp,
    )
    tmp.replace(path)


def load(path: Path, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build(ck["model"], len(ck["classes"]), ck["img_size"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
