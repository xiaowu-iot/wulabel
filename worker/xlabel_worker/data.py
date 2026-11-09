"""读取桌面端导出的 COCO 数据集（检测用框，分割另外读多边形生成掩膜）"""

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
    def __init__(self, root: Path, split: str, train: bool, seg: bool = False):
        root = Path(root)
        doc = json.loads((root / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
        self.dir = root / "images" / split
        self.images = doc["images"]
        self.train = train
        self.seg = seg
        self.anns: dict[int, list] = {}
        for a in doc["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def target(self, i: int) -> dict:
        im = self.images[i]
        w, h = int(im.get("width") or 1), int(im.get("height") or 1)
        boxes, labels, masks = [], [], []
        for a in self.anns.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            if bw >= 1 and bh >= 1:
                boxes.append([x, y, x + bw, y + bh])
                labels.append(a["category_id"])  # 1..n，0 是背景
                if self.seg:
                    masks.append(_rasterize(a.get("segmentation"), a["bbox"], w, h))
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
        return t

    def __getitem__(self, i: int):
        im = self.images[i]
        img = load_image(self.dir / im["file_name"])
        tgt = self.target(i)
        tgt["image_id"] = im["id"]
        t = F.to_tensor(img)
        if self.train and random.random() < 0.5:
            t = t.flip(-1)
            w = t.shape[-1]
            b = tgt["boxes"]
            tgt["boxes"] = b[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
            if "masks" in tgt and tgt["masks"].numel():
                tgt["masks"] = tgt["masks"].flip(-1)
        return t, tgt


def collate(batch):
    return tuple(zip(*batch))


def has_split(root: Path, split: str) -> bool:
    f = Path(root) / "annotations" / f"instances_{split}.json"
    if not f.exists():
        return False
    return len(json.loads(f.read_text(encoding="utf-8"))["images"]) > 0
