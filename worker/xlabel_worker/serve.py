"""常驻推理服务：从标准输入逐行读请求，结果逐行写到标准输出。每个请求最后都以 done 或 error 结束。

请求：
    {"cmd": "ping"}
    {"cmd": "predict", "weights": "...best.pt", "conf": 0.3, "images": [{"id": 1, "path": "..."}]}
    {"cmd": "sam", "path": "...", "points": [[x, y, 1/0], ...], "box": [x0, y0, x1, y1] 或 null}
        → {"type": "sam", "points": [多边形], "box": [x, y, w, h], "score": s}
    {"cmd": "text", "queries": [[类别序号, "英文提示词"], ...], "conf": 0.2, "seg": false, "images": [...]}
        → 每张图一行 pred（boxes；seg=true 时另有 SAM 出的 polys）
    {"cmd": "sam_boxes", "images": [{"id", "path", "boxes": [[x0, y0, x1, y1, 类别序号, 分数], ...]}]}
        → 每张图一行 pred（polys），把别处得到的框转成分割轮廓
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
from .assist import mask_to_poly
from .data import cls_preprocess, load_image
from .protocol import emit


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
        key = (str(p), p.stat().st_mtime, p.with_suffix(".json").stat().st_mtime if p.suffix == ".onnx" else 0)
        if key != self.key:
            if p.suffix.lower() == ".onnx":
                # 导入的外部模型（YOLO 等）
                from . import onnx_model

                self.model, info = onnx_model.load(p)
                self.ck = {"classes": info.get("classes", []), "model": "external", "use": info.get("use")}
                self.task = "external"
            else:
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
                if self.task == "external":
                    use = self.ck.get("use")
                    boxes = [
                        [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2), c, round(s, 4)]
                        for x0, y0, x1, y1, c, s in self.model.predict(img, conf)
                        if use is None or c in use
                    ]
                    emit("pred", id=it["id"], boxes=boxes, polys=[], poses=[], tags=[])
                    n += 1
                    continue
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

    # ---------- 标注辅助：SAM、文本提示 ----------

    def _sam(self):
        if getattr(self, "sam_model", None) is None:
            from .assist import Sam

            self.sam_model = Sam(self.device)
        return self.sam_model

    def _owl(self):
        if getattr(self, "owl_model", None) is None:
            from .assist import Owl

            self.owl_model = Owl(self.device)
        return self.owl_model

    def sam(self, req: dict):
        r = self._sam().segment(req["path"], req.get("points") or [], req.get("box"))
        emit("sam", **r)
        emit("done")

    def text(self, req: dict):
        """文本提示找目标；seg=true 时再用 SAM 把框变成轮廓"""
        queries = req["queries"]  # [[类别序号, 英文], ...]
        texts = [q[1] for q in queries]
        cls_of = [int(q[0]) for q in queries]
        conf = float(req.get("conf", 0.2))
        seg = bool(req.get("seg"))
        owl = self._owl()
        n = 0
        for it in req["images"]:
            try:
                dets = owl.detect(it["path"], texts, conf)
                boxes = [
                    [round(b[0], 2), round(b[1], 2), round(b[2] - b[0], 2), round(b[3] - b[1], 2), cls_of[q], round(s, 4)]
                    for b, q, s in dets
                ]
                polys = []
                if seg and dets:
                    for (b, q, s), r in zip(dets, self._sam().segment_boxes(it["path"], [d[0] for d in dets])):
                        if r["points"]:
                            polys.append([cls_of[q], round(s, 4)] + r["points"])
                emit("pred", id=it["id"], boxes=boxes, polys=polys, poses=[], tags=[])
            except Exception as e:
                emit("pred", id=it["id"], boxes=[], polys=[], poses=[], tags=[], error=str(e))
            n += 1
        emit("done", count=n)

    def sam_boxes(self, req: dict):
        """把已有的框（文本提示、巡瞳 AI 等给的）用 SAM 转成分割轮廓"""
        n = 0
        for it in req["images"]:
            try:
                bs = it.get("boxes") or []
                rs = self._sam().segment_boxes(it["path"], [b[:4] for b in bs])
                polys = [[int(b[4]), round(float(b[5]), 4)] + r["points"] for b, r in zip(bs, rs) if r["points"]]
                emit("pred", id=it["id"], boxes=[], polys=polys, poses=[], tags=[])
            except Exception as e:
                emit("pred", id=it["id"], boxes=[], polys=[], poses=[], tags=[], error=str(e))
            n += 1
        emit("done", count=n)


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
            elif cmd == "sam":
                p.sam(req)
            elif cmd == "text":
                p.text(req)
            elif cmd == "sam_boxes":
                p.sam_boxes(req)
            elif cmd == "quit":
                break
            else:
                emit("error", msg=f"未知命令：{cmd}")
        except Exception as e:
            emit("error", msg=str(e))
