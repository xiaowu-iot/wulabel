"""常驻推理服务：从标准输入逐行读请求，结果逐行写到标准输出。

请求：
    {"cmd": "ping"}
    {"cmd": "predict", "weights": "...best.pt", "conf": 0.3, "images": [{"id": 1, "path": "..."}]}
响应：
    {"type": "pong", ...}
    {"type": "pred", "id": 1, "boxes": [[x, y, w, h, 类别序号, 分数], ...],
     "polys": [[类别序号, 分数, x1, y1, x2, y2, ...], ...],          分割模型才有
     "poses": [[类别序号, 分数, x, y, w, h, kx1, ky1, kv1, ...], ...], 关键点模型才有（kv：2 找到、0 没找到）
     "tags": [[类别序号, 分数], ...]}                                  分类模型才有，前 5 个，分数从高到低
    每张图一行
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
from .data import cls_preprocess, load_image
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


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Predictor:
    def __init__(self):
        self.device = pick_device()
        self.key = None
        self.model = None
        self.ck = None
        self.task = "detect"

    def ensure(self, weights: str):
        p = Path(weights)
        key = (str(p), p.stat().st_mtime)
        if key != self.key:
            self.model, self.ck = models.load(p, self.device)
            self.task = models.task_of(self.ck.get("model", ""))
            self.key = key

    def _classify(self, img) -> list:
        x = cls_preprocess(img, int(self.ck["img_size"])).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.model(x).float(), dim=1)[0].cpu()
        k = min(5, prob.numel())
        top = torch.topk(prob, k)
        return [[int(i), round(float(s), 4)] for s, i in zip(top.values, top.indices)]

    @torch.no_grad()
    def predict(self, weights: str, items: list[dict], conf: float):
        self.ensure(weights)
        n = 0
        for it in items:
            try:
                img = load_image(Path(it["path"]))
                if self.task == "classify":
                    # 分类：前 5 个类别都返回，由桌面端按置信度决定要不要采用
                    emit("pred", id=it["id"], boxes=[], polys=[], poses=[], tags=self._classify(img))
                    n += 1
                    continue
                out = self.model([F.to_tensor(img).to(self.device)])[0]
                masks = out["masks"].cpu().numpy() if (self.task == "segment" and "masks" in out) else None
                kps = out["keypoints"].cpu().numpy() if (self.task == "pose" and "keypoints" in out) else None
                kscores = out["keypoints_scores"].cpu().numpy() if kps is not None and "keypoints_scores" in out else None
                boxes, polys, poses = [], [], []
                data = zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist())
                for i, (b, s, l) in enumerate(data):
                    if s < conf:
                        continue
                    x0, y0, x1, y1 = b
                    box = [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]
                    boxes.append(box + [l - 1, round(s, 4)])
                    if masks is not None:
                        poly = mask_to_poly(masks[i, 0])
                        if poly:
                            polys.append([l - 1, round(s, 4)] + poly)
                    if kps is not None:
                        flat = []
                        for j, (kx, ky, _) in enumerate(kps[i]):
                            # 关键点热图分数 > 0 当成找到了（标成可见），否则当成没找到
                            if kscores is None or kscores[i, j] > 0:
                                flat += [round(float(kx), 1), round(float(ky), 1), 2]
                            else:
                                flat += [0, 0, 0]
                        poses.append([l - 1, round(s, 4)] + box + flat)
                emit("pred", id=it["id"], boxes=boxes, polys=polys, poses=poses, tags=[])
            except Exception as e:
                emit("pred", id=it["id"], boxes=[], polys=[], poses=[], tags=[], error=str(e))
            n += 1
        emit("done", count=n, classes=self.ck["classes"], task=self.task)


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
