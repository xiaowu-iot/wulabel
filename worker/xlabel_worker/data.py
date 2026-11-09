"""读取桌面端导出的 COCO 数据集"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F


def load_image(path: Path) -> Image.Image:
    # 按 EXIF 摆正，和桌面端记录的宽高一致
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, train: bool):
        root = Path(root)
        doc = json.loads((root / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
        self.dir = root / "images" / split
        self.images = doc["images"]
        self.train = train
        self.anns: dict[int, list] = {}
        for a in doc["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def target(self, i: int):
        im = self.images[i]
        boxes, labels = [], []
        for a in self.anns.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w >= 1 and h >= 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(a["category_id"])  # 1..n，0 是背景
        return (
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(labels, dtype=torch.int64),
        )

    def __getitem__(self, i: int):
        im = self.images[i]
        img = load_image(self.dir / im["file_name"])
        boxes, labels = self.target(i)
        t = F.to_tensor(img)
        if self.train and random.random() < 0.5:
            t = t.flip(-1)
            w = t.shape[-1]
            boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
        return t, {"boxes": boxes, "labels": labels, "image_id": im["id"]}


def collate(batch):
    return tuple(zip(*batch))


def has_split(root: Path, split: str) -> bool:
    f = Path(root) / "annotations" / f"instances_{split}.json"
    if not f.exists():
        return False
    return len(json.loads(f.read_text(encoding="utf-8"))["images"]) > 0
