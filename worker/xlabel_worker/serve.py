"""常驻推理服务：从标准输入逐行读请求，结果逐行写到标准输出。

请求：
    {"cmd": "ping"}
    {"cmd": "predict", "weights": "...best.pt", "conf": 0.3, "images": [{"id": 1, "path": "..."}]}
响应：
    {"type": "pong", ...}
    {"type": "pred", "id": 1, "boxes": [[x, y, w, h, 类别序号, 分数], ...]}   每张图一行
    {"type": "done", "count": n}
    {"type": "error", "msg": "..."}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torchvision.transforms import functional as F

from . import models
from .data import load_image
from .protocol import emit


class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.key = None
        self.model = None
        self.ck = None

    def ensure(self, weights: str):
        p = Path(weights)
        key = (str(p), p.stat().st_mtime)
        if key != self.key:
            self.model, self.ck = models.load(p, self.device)
            self.key = key

    @torch.no_grad()
    def predict(self, weights: str, items: list[dict], conf: float):
        self.ensure(weights)
        n = 0
        for it in items:
            try:
                img = load_image(Path(it["path"]))
                out = self.model([F.to_tensor(img).to(self.device)])[0]
                boxes = []
                for b, s, l in zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()):
                    if s >= conf:
                        x0, y0, x1, y1 = b
                        boxes.append([round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2), l - 1, round(s, 4)])
                emit("pred", id=it["id"], boxes=boxes)
            except Exception as e:
                emit("pred", id=it["id"], boxes=[], error=str(e))
            n += 1
        emit("done", count=n, classes=self.ck["classes"])


def run() -> None:
    p = Predictor()
    for line in sys.stdin:
        line = line.strip().lstrip("﻿")
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "ping":
                emit("pong", device=str(p.device))
            elif cmd == "predict":
                p.predict(req["weights"], req["images"], float(req.get("conf", 0.3)))
            elif cmd == "quit":
                break
            else:
                emit("error", msg=f"未知命令：{cmd}")
        except Exception as e:
            emit("error", msg=str(e))
