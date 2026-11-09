"""内置模型（torchvision，BSD 许可）

task：detect 目标检测 / segment 实例分割 / pose 关键点 / classify 图像分类
kind：rcnn 两阶段（Faster/Mask/Keypoint R-CNN）、retinanet / fcos / ssd 单阶段检测、cls 分类
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from .protocol import log

# 名称和说明给界面展示（界面另有一份，见桌面端 Train.tsx 的 MODELS）
MODELS = {
    # ---------- 目标检测：两阶段 ----------
    "frcnn_mobilenet": {
        "name": "Faster R-CNN MobileNetV3",
        "builder": "fasterrcnn_mobilenet_v3_large_fpn",
        "task": "detect",
        "kind": "rcnn",
    },
    "frcnn_resnet50": {
        "name": "Faster R-CNN ResNet50 v2",
        "builder": "fasterrcnn_resnet50_fpn_v2",
        "task": "detect",
        "kind": "rcnn",
    },
    # ---------- 目标检测：单阶段（没有 RoIAlign，转 TensorRT / NPU 更顺） ----------
    "retinanet_r50": {
        "name": "RetinaNet ResNet50 v2",
        "builder": "retinanet_resnet50_fpn_v2",
        "task": "detect",
        "kind": "retinanet",
    },
    "fcos_r50": {
        "name": "FCOS ResNet50",
        "builder": "fcos_resnet50_fpn",
        "task": "detect",
        "kind": "fcos",
    },
    "ssdlite": {
        "name": "SSDLite MobileNetV3（320）",
        "builder": "ssdlite320_mobilenet_v3_large",
        "task": "detect",
        "kind": "ssd",
    },
    # ---------- 实例分割 ----------
    "maskrcnn_r50": {
        "name": "Mask R-CNN ResNet50",
        "builder": "maskrcnn_resnet50_fpn",
        "task": "segment",
        "kind": "rcnn",
    },
    "maskrcnn_r50_v2": {
        "name": "Mask R-CNN ResNet50 v2",
        "builder": "maskrcnn_resnet50_fpn_v2",
        "task": "segment",
        "kind": "rcnn",
    },
    # ---------- 关键点 ----------
    "keypointrcnn_r50": {
        "name": "Keypoint R-CNN ResNet50",
        "builder": "keypointrcnn_resnet50_fpn",
        "task": "pose",
        "kind": "rcnn",
    },
    # ---------- 图像分类 ----------
    "cls_mobilenet": {
        "name": "MobileNetV3 Large",
        "builder": "mobilenet_v3_large",
        "task": "classify",
        "kind": "cls",
    },
    "cls_efficientnet": {
        "name": "EfficientNet-B0",
        "builder": "efficientnet_b0",
        "task": "classify",
        "kind": "cls",
    },
    "cls_resnet50": {
        "name": "ResNet50",
        "builder": "resnet50",
        "task": "classify",
        "kind": "cls",
    },
}

# ImageNet 均值方差（分类模型的输入归一化）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def task_of(key: str) -> str:
    return MODELS.get(key, {}).get("task", "detect")


def kind_of(key: str) -> str:
    return MODELS.get(key, {}).get("kind", "rcnn")


def is_seg(key: str) -> bool:
    return task_of(key) == "segment"


def is_pose(key: str) -> bool:
    return task_of(key) == "pose"


def is_cls(key: str) -> bool:
    return task_of(key) == "classify"


def _with_weights(builder, pretrained: bool, extra: dict, label: str):
    """先试完整预训练权重，再试只用骨干网络权重，都不行就随机初始化"""
    if pretrained:
        tries = [(dict(weights="DEFAULT"), f"{label}预训练权重")]
        if label != "ImageNet ":
            tries.append((dict(weights=None, weights_backbone="DEFAULT"), "骨干网络预训练权重"))
        for kw, what in tries:
            try:
                m = builder(**kw, **extra)
                log(f"已加载 {what}")
                return m
            except Exception as e:  # 多半是网络问题
                log(f"加载 {what} 失败：{e}")
        log("没有可用的预训练权重，从头训练，效果会差很多")
    if label == "ImageNet ":
        return builder(weights=None, **extra)
    return builder(weights=None, weights_backbone=None, **extra)


def _build_cls(key: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    from torchvision import models as M

    model = _with_weights(getattr(M, MODELS[key]["builder"]), pretrained, {}, "ImageNet ")
    # 把最后一个全连接层换成我们的类别数
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        last = model.classifier[-1]
        model.classifier[-1] = torch.nn.Linear(last.in_features, num_classes)
    return model


def build(key: str, num_classes: int, img_size: int, pretrained: bool = True, num_keypoints: int = 17) -> torch.nn.Module:
    """num_classes 不含背景。预训练权重下载失败时退回只用骨干网络权重，再不行就随机初始化。"""
    if key not in MODELS:
        raise ValueError(f"不支持的模型：{key}")
    kind = kind_of(key)
    if kind == "cls":
        return _build_cls(key, num_classes, pretrained)

    from torchvision.models import detection as D

    builder = getattr(D, MODELS[key]["builder"])
    n = num_classes + 1  # 检测类模型多一个背景类（序号 0）
    # SSDLite 固定 320 输入，不接受尺寸参数；其他检测器按训练尺寸缩放
    size = {} if kind == "ssd" else dict(min_size=img_size, max_size=int(img_size * 1.667))
    model = _with_weights(builder, pretrained, size, "COCO ")

    if kind == "rcnn":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)
        if task_of(key) == "segment":
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

            in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, n)
        if task_of(key) == "pose" and num_keypoints != 17:
            # COCO 预训练的是 17 个人体关键点；点数不同时换一个新的关键点头
            from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

            in_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
            model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_kp, num_keypoints)
    elif kind == "retinanet":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        head = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels, head.num_anchors, n, norm_layer=partial(torch.nn.GroupNorm, 32)
        )
    elif kind == "fcos":
        from torchvision.models.detection.fcos import FCOSClassificationHead

        head = model.head.classification_head
        model.head.classification_head = FCOSClassificationHead(model.backbone.out_channels, head.num_anchors, n)
    elif kind == "ssd":
        from torchvision.models.detection import _utils as det_utils
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

        in_ch = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_ch, anchors, n, partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        )
    return model


def save(path: Path, model: torch.nn.Module, key: str, classes: list[str], img_size: int, extra: dict | None = None):
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "xlabel-det-1",
            "model": key,
            "task": task_of(key),
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
    model = build(
        ck["model"], len(ck["classes"]), ck["img_size"], pretrained=False, num_keypoints=int(ck.get("num_keypoints", 17))
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
