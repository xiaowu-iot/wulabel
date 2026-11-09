"""训练数据增强。训练和「效果预览」共用这一份代码，预览看到的就是训练时真实用的。

配置（训练参数里的 aug，缺的项用默认值）：
    hflip       水平翻转（关键点按左右对应关系交换）
    vflip       上下翻转（关键点项目不用）
    brightness  亮度随机变化幅度，0.2 表示 ±20%
    contrast    对比度随机变化幅度
    saturation  饱和度随机变化幅度
    hue         色调随机偏移（0~0.5）
    blur        随机模糊的概率
    noise       随机噪声强度（0~0.1）
    scale       多尺度训练幅度，0.2 表示输入尺寸在 ±20% 之间随机（检测类模型，SSDLite 不支持）
    rotate      随机旋转角度（只用于图像分类）
    crop        随机裁剪时最少保留的面积比例（只用于图像分类）
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

DEFAULTS = {
    # 和以前的行为一致：检测只做水平翻转
    "detect": {"hflip": True, "vflip": False, "brightness": 0, "contrast": 0, "saturation": 0, "hue": 0, "blur": 0, "noise": 0, "scale": 0},
    # 分类以前就有随机裁剪、翻转和轻微的亮度对比度变化
    "classify": {
        "hflip": True, "vflip": False, "brightness": 0.2, "contrast": 0.2, "saturation": 0, "hue": 0,
        "blur": 0, "noise": 0, "rotate": 0, "crop": 0.5,
    },
}


def config(aug: dict | None, task: str) -> dict:
    base = DEFAULTS["classify" if task == "classify" else "detect"]
    return {**base, **{k: v for k, v in (aug or {}).items() if v is not None}}


def _jitter(amount: float) -> float:
    return random.uniform(max(0.0, 1 - amount), 1 + amount)


def photometric(t: torch.Tensor, c: dict) -> torch.Tensor:
    """只改像素颜色、不改几何：对检测框、轮廓、关键点都安全。t 是 [3, H, W]、0~1"""
    if c.get("brightness"):
        t = F.adjust_brightness(t, _jitter(c["brightness"]))
    if c.get("contrast"):
        t = F.adjust_contrast(t, _jitter(c["contrast"]))
    if c.get("saturation"):
        t = F.adjust_saturation(t, _jitter(c["saturation"]))
    if c.get("hue"):
        t = F.adjust_hue(t, random.uniform(-min(c["hue"], 0.5), min(c["hue"], 0.5)))
    if c.get("blur") and random.random() < c["blur"]:
        k = random.choice([3, 5])
        t = F.gaussian_blur(t, [k, k], [random.uniform(0.3, 1.5)] * 2)
    if c.get("noise"):
        t = t + torch.randn_like(t) * random.uniform(0, c["noise"])
    return t.clamp(0, 1)


def detection(t: torch.Tensor, tgt: dict, c: dict, flip_index: list[int] | None = None, pose: bool = False):
    """检测 / 分割 / 关键点：翻转（同步改框、掩膜、关键点）+ 颜色变化"""
    h, w = t.shape[-2:]
    # 关键点没有左右对应关系时不能水平翻转（翻转后「左眼」会跑到右边）
    if c.get("hflip") and (not pose or flip_index) and random.random() < 0.5:
        t = t.flip(-1)
        b = tgt["boxes"]
        tgt["boxes"] = b[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
        if "masks" in tgt and tgt["masks"].numel():
            tgt["masks"] = tgt["masks"].flip(-1)
        if "keypoints" in tgt and tgt["keypoints"].numel():
            k = tgt["keypoints"].clone()
            vis = k[..., 2] > 0
            k[..., 0] = torch.where(vis, w - k[..., 0], k[..., 0])
            tgt["keypoints"] = k[:, flip_index] if flip_index else k
    if c.get("vflip") and not pose and random.random() < 0.5:
        t = t.flip(-2)
        b = tgt["boxes"]
        tgt["boxes"] = b[:, [0, 3, 2, 1]] * torch.tensor([1, -1, 1, -1]) + torch.tensor([0, h, 0, h])
        if "masks" in tgt and tgt["masks"].numel():
            tgt["masks"] = tgt["masks"].flip(-2)
    return photometric(t, c), tgt


def classify(img: Image.Image, size: int, c: dict) -> Image.Image:
    """图像分类：随机裁剪缩放、旋转、翻转、颜色变化，输出 size×size 的图（还没归一化）"""
    if c.get("rotate"):
        img = img.rotate(random.uniform(-c["rotate"], c["rotate"]), resample=Image.BILINEAR, fillcolor=(114, 114, 114))
    lo = min(max(float(c.get("crop") or 1.0), 0.05), 1.0)
    i0, j0, hh, ww = _crop_params(img, (lo, 1.0))
    img = F.resized_crop(img, i0, j0, hh, ww, [size, size], antialias=True)
    if c.get("hflip") and random.random() < 0.5:
        img = F.hflip(img)
    if c.get("vflip") and random.random() < 0.5:
        img = F.vflip(img)
    return F.to_pil_image(photometric(F.to_tensor(img), c))


def _crop_params(img: Image.Image, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)):
    w, h = img.size
    area = w * h
    for _ in range(10):
        a = area * random.uniform(*scale)
        r = np.exp(random.uniform(np.log(ratio[0]), np.log(ratio[1])))
        cw = int(round((a * r) ** 0.5))
        ch = int(round((a / r) ** 0.5))
        if 0 < cw <= w and 0 < ch <= h:
            return random.randint(0, h - ch), random.randint(0, w - cw), ch, cw
    return 0, 0, h, w


def multiscale_sizes(img_size: int, scale: float) -> tuple[int, ...]:
    """多尺度训练的候选尺寸（32 的倍数），最后一个是原尺寸——评估时 torchvision 用最后一个"""
    if not scale:
        return (img_size,)
    sizes = {max(64, int(round(img_size * f / 32)) * 32) for f in np.linspace(1 - scale, 1 + scale, 5)}
    return tuple(sorted(sizes - {img_size})) + (img_size,)


# ---------- 效果预览 ----------


def preview(cfg: dict) -> None:
    """cfg: task, img_size, aug, n（每张图出几张）, images: [{path, boxes: [[x, y, w, h]]}]"""
    import base64
    import io

    from PIL import ImageDraw

    from .data import load_image
    from .protocol import emit

    task = cfg.get("task", "detect")
    c = config(cfg.get("aug"), task)
    size = int(cfg.get("img_size", 224 if task == "classify" else 640))
    n = int(cfg.get("n", 2))
    for it in cfg["images"]:
        try:
            src = load_image(it["path"])
        except Exception:
            continue
        for _ in range(n):
            if task == "classify":
                out = classify(src, size, c)
                draw_boxes = []
            else:
                boxes = [[x, y, x + bw, y + bh] for x, y, bw, bh in it.get("boxes", [])]
                tgt = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)}
                t, tgt = detection(F.to_tensor(src), tgt, c, pose=task == "pose")
                out = F.to_pil_image(t)
                draw_boxes = tgt["boxes"].tolist()
            s = 300 / max(out.size)
            small = out.resize((max(1, int(out.width * s)), max(1, int(out.height * s))))
            d = ImageDraw.Draw(small)
            for x0, y0, x1, y1 in draw_boxes:
                d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(255, 64, 64), width=2)
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=80)
            emit("preview", img=base64.b64encode(buf.getvalue()).decode())
    emit("done")
