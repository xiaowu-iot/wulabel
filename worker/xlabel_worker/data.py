"""读取桌面端导出的训练数据集。

- 检测 / 分割 / 关键点：COCO 格式（annotations/instances_{split}.json + images/{split}/）
  分割另外把多边形画成掩膜，关键点读 keypoints 字段
- 图像分类：分类文件夹 + 清单（annotations/{split}.json，{"classes": [...], "images": [{"id", "file", "label"}]}）
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import functional as F


def load_image(path: Path) -> Image.Image:
    # 按 EXIF 摆正，和桌面端记录的宽高一致
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _rasterize(seg, bbox, w: int, h: int) -> np.ndarray:
    """一个实例的分割多边形画成 0/1 掩膜；没有多边形就用检测框填充。"""
    m = Image.new("L", (int(w), int(h)), 0)
    d = ImageDraw.Draw(m)
    drawn = False
    for poly in seg or []:
        if isinstance(poly, (list, tuple)) and len(poly) >= 6:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
            d.polygon(pts, fill=1)
            drawn = True
    if not drawn:
        x, y, bw, bh = bbox
        d.rectangle([x, y, x + bw, y + bh], fill=1)
    return np.asarray(m, dtype=np.uint8)


class CocoDetection(torch.utils.data.Dataset):
    """seg：同时给出掩膜；num_keypoints > 0：同时给出关键点 [N, K, 3]（x, y, 可见性）。
    flip_index：左右翻转时关键点的对应关系（左眼 ↔ 右眼），没有就不做翻转增强。"""

    def __init__(
        self,
        root: Path,
        split: str,
        train: bool,
        seg: bool = False,
        num_keypoints: int = 0,
        flip_index: list[int] | None = None,
    ):
        root = Path(root)
        doc = json.loads((root / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
        self.dir = root / "images" / split
        self.images = doc["images"]
        self.train = train
        self.seg = seg
        self.nk = num_keypoints
        self.flip_index = flip_index if (flip_index and len(flip_index) == num_keypoints) else None
        self.anns: dict[int, list] = {}
        for a in doc["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def target(self, i: int) -> dict:
        im = self.images[i]
        w, h = int(im.get("width") or 1), int(im.get("height") or 1)
        boxes, labels, masks, kps = [], [], [], []
        for a in self.anns.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            if bw >= 1 and bh >= 1:
                boxes.append([x, y, x + bw, y + bh])
                labels.append(a["category_id"])  # 1..n，0 是背景
                if self.seg:
                    masks.append(_rasterize(a.get("segmentation"), a["bbox"], w, h))
                if self.nk:
                    k = list(a.get("keypoints") or [])
                    k = (k + [0.0] * (self.nk * 3))[: self.nk * 3]
                    kps.append(np.array(k, dtype=np.float32).reshape(self.nk, 3))
        t = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        if self.seg:
            t["masks"] = (
                torch.as_tensor(np.array(masks, dtype=np.uint8))
                if masks
                else torch.zeros((0, h, w), dtype=torch.uint8)
            )
        if self.nk:
            t["keypoints"] = (
                torch.as_tensor(np.array(kps, dtype=np.float32))
                if kps
                else torch.zeros((0, self.nk, 3), dtype=torch.float32)
            )
        return t

    def __getitem__(self, i: int):
        im = self.images[i]
        img = load_image(self.dir / im["file_name"])
        tgt = self.target(i)
        tgt["image_id"] = im["id"]
        t = F.to_tensor(img)
        # 关键点没有左右对应关系时不能翻转（翻转后「左眼」会跑到右边）
        can_flip = not self.nk or self.flip_index is not None
        if self.train and can_flip and random.random() < 0.5:
            t = t.flip(-1)
            w = t.shape[-1]
            b = tgt["boxes"]
            tgt["boxes"] = b[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
            if "masks" in tgt and tgt["masks"].numel():
                tgt["masks"] = tgt["masks"].flip(-1)
            if "keypoints" in tgt and tgt["keypoints"].numel():
                k = tgt["keypoints"].clone()
                vis = k[..., 2] > 0
                k[..., 0] = torch.where(vis, w - k[..., 0], k[..., 0])
                tgt["keypoints"] = k[:, self.flip_index]
        return t, tgt


class ClsFolder(torch.utils.data.Dataset):
    """图像分类：读清单 annotations/{split}.json，图片在数据集目录下的 file 路径"""

    def __init__(self, root: Path, split: str, train: bool, size: int):
        root = Path(root)
        doc = json.loads((root / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
        self.root = root
        self.items = doc["images"]
        self.train = train
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        from . import models

        it = self.items[i]
        img = load_image(self.root / it["file"])
        s = self.size
        if self.train:
            # 随机裁剪缩放 + 水平翻转 + 轻微颜色扰动
            i0, j0, hh, ww = _random_resized_crop_params(img, scale=(0.5, 1.0))
            img = F.resized_crop(img, i0, j0, hh, ww, [s, s], antialias=True)
            if random.random() < 0.5:
                img = F.hflip(img)
            img = F.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = F.adjust_contrast(img, random.uniform(0.8, 1.2))
        else:
            img = F.resize(img, [s, s], antialias=True)
        t = F.normalize(F.to_tensor(img), models.IMAGENET_MEAN, models.IMAGENET_STD)
        return t, int(it["label"]), int(it["id"])


def _random_resized_crop_params(img: Image.Image, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)):
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


def cls_preprocess(img: Image.Image, size: int) -> torch.Tensor:
    """分类推理：整图缩放到 size×size，ImageNet 归一化"""
    from . import models

    img = F.resize(img, [size, size], antialias=True)
    return F.normalize(F.to_tensor(img), models.IMAGENET_MEAN, models.IMAGENET_STD)


def collate(batch):
    return tuple(zip(*batch))


def has_split(root: Path, split: str) -> bool:
    f = Path(root) / "annotations" / f"instances_{split}.json"
    if not f.exists():
        return False
    return len(json.loads(f.read_text(encoding="utf-8"))["images"]) > 0


def has_cls_split(root: Path, split: str) -> bool:
    f = Path(root) / "annotations" / f"{split}.json"
    if not f.exists():
        return False
    return len(json.loads(f.read_text(encoding="utf-8"))["images"]) > 0
