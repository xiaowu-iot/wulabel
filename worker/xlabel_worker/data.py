"""读取桌面端导出的训练数据集。

- 检测 / 分割 / 关键点：COCO 格式（annotations/instances_{split}.json + images/{split}/）
  分割另外把多边形画成掩膜，关键点读 keypoints 字段
- 图像分类：分类文件夹 + 清单（annotations/{split}.json，{"classes": [...], "images": [{"id", "file", "label"}]}）
"""

from __future__ import annotations

import json
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
        aug: dict | None = None,
    ):
        from . import augment

        self.aug = augment.config(aug, "detect")
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
        if self.train:
            from . import augment

            t, tgt = augment.detection(t, tgt, self.aug, self.flip_index, pose=bool(self.nk))
        return t, tgt


class ClsFolder(torch.utils.data.Dataset):
    """图像分类：读清单 annotations/{split}.json，图片在数据集目录下的 file 路径"""

    def __init__(self, root: Path, split: str, train: bool, size: int, aug: dict | None = None):
        from . import augment

        root = Path(root)
        doc = json.loads((root / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
        self.root = root
        self.items = doc["images"]
        self.train = train
        self.size = size
        self.aug = augment.config(aug, "classify")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        from . import models

        it = self.items[i]
        img = load_image(self.root / it["file"])
        s = self.size
        if self.train:
            from . import augment

            img = augment.classify(img, s, self.aug)
        else:
            img = F.resize(img, [s, s], antialias=True)
        t = F.normalize(F.to_tensor(img), models.IMAGENET_MEAN, models.IMAGENET_STD)
        return t, int(it["label"]), int(it["id"])


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
