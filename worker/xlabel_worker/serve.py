"""常驻推理服务：从标准输入逐行读请求，结果逐行写到标准输出。

请求：
    {"cmd": "ping"}
    {"cmd": "predict", "weights": "...best.pt", "conf": 0.3, "images": [{"id": 1, "path": "..."}]}
响应：
    {"type": "pong", ...}
    {"type": "pred", "id": 1, "boxes": [[x, y, w, h, 类别序号, 分数], ...],
     "polys": [[类别序号, 分数, x1, y1, x2, y2, ...], ...]}   分割模型才有 polys，每张图一行
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


def mask_to_poly(mask, thresh: float = 0.5):
    """把一个实例的概率掩膜转成多边形点 [x1, y1, x2, y2, ...]；失败返回 None。需要 opencv。"""
    try:
        import cv2
    except Exception:
        return None
    m = (mask > thresh).astype("uint8")
    if int(m.sum()) < 10:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return None
    eps = 0.004 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        pts = c.reshape(-1, 2)
        approx = pts[:: max(1, len(pts) // 24)]
        if len(approx) < 3:
            return None
    out = []
    for x, y in approx:
        out.append(round(float(x), 1))
        out.append(round(float(y), 1))
    return out


class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.key = None
        self.model = None
        self.ck = None
        self.seg = False

    def ensure(self, weights: str):
        p = Path(weights)
        key = (str(p), p.stat().st_mtime)
        if key != self.key:
            self.model, self.ck = models.load(p, self.device)
            self.seg = models.is_seg(self.ck.get("model", ""))
            self.key = key

    @torch.no_grad()
    def predict(self, weights: str, items: list[dict], conf: float):
        self.ensure(weights)
        n = 0
        for it in items:
            try:
                img = load_image(Path(it["path"]))
                out = self.model([F.to_tensor(img).to(self.device)])[0]
                masks = out["masks"].cpu().numpy() if (self.seg and "masks" in out) else None
                boxes, polys = [], []
                data = zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist())
                for i, (b, s, l) in enumerate(data):
                    if s < conf:
                        continue
                    x0, y0, x1, y1 = b
                    boxes.append([round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2), l - 1, round(s, 4)])
                    if masks is not None:
                        poly = mask_to_poly(masks[i, 0])
                        if poly:
                            polys.append([l - 1, round(s, 4)] + poly)
                emit("pred", id=it["id"], boxes=boxes, polys=polys)
            except Exception as e:
                emit("pred", id=it["id"], boxes=[], polys=[], error=str(e))
            n += 1
        emit("done", count=n, classes=self.ck["classes"], seg=self.seg)


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
