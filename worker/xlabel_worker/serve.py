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
import os
import sys
from pathlib import Path

import torch
from torchvision.transforms import functional as F

from . import models
from .assist import mask_to_poly
from .data import cls_preprocess, load_image
from .i18n import tr
from .protocol import emit


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def onnx_pred(model, img, conf: float, use=None) -> dict:
    """ONNX 模型的结果换成 pred 行的格式（boxes / polys / poses / tags）；use：只要这些类别（None 表示全要）"""
    boxes, polys, poses, tags = [], [], [], []
    for d in model.predict_full(img, conf):
        c, s = d["cls"], round(d["score"], 4)
        if use is not None and c not in use:
            continue
        if "box" not in d:
            tags.append([c, s])
            continue
        x0, y0, x1, y1 = d["box"]
        box = [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]
        boxes.append(box + [c, s])
        if d.get("polygon"):
            polys.append([c, s] + [round(v, 1) for v in d["polygon"]])
        if "keypoints" in d:
            kp = d["keypoints"]
            poses.append([c, s] + box + [round(v, 1) if j % 3 < 2 else int(v) for j, v in enumerate(kp)])
    return {"boxes": boxes, "polys": polys, "poses": poses, "tags": tags}


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
                # ONNX 模型（导入的外部模型、Ultralytics 等引擎训练后导出的）
                from . import onnx_model

                self.model, info = onnx_model.load(p)
                self.ck = {"classes": info.get("classes", []), "model": "onnx", "use": info.get("use")}
                self.task = "onnx"
                self.onnx_task = self.model.task
            else:
                self.model, self.ck = models.load(p, self.device)
                self.task = models.task_of(self.ck.get("model", ""))
            self.key = key

    def _classify(self, img) -> list:
        x = cls_preprocess(img, int(self.ck["img_size"])).unsqueeze(0).to(self.device)
        logits = self.model(x).float()
        if self.ck.get("multi_label"):
            # 多标签：每个类别各自的分数，按分数从高到低都返回（太低的不要），由桌面端按阈值决定采用哪些
            prob = torch.sigmoid(logits)[0].cpu()
            order = torch.argsort(prob, descending=True)
            return [[int(i), round(float(prob[i]), 4)] for i in order[:20] if float(prob[i]) >= 0.01]
        prob = torch.softmax(logits, dim=1)[0].cpu()
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
                if self.task == "onnx":
                    emit("pred", id=it["id"], **onnx_pred(self.model, img, conf, self.ck.get("use")))
                    n += 1
                    continue
                if self.task == "classify":
                    # 分类：前 5 个类别（多标签是分数够的全部类别）都返回，由桌面端按置信度决定要不要采用
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
        emit("done", count=n, classes=self.ck["classes"], task=getattr(self, "onnx_task", self.task) if self.task == "onnx" else self.task)

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


_FONTS: dict[int, object] = {}


def _font_paths() -> list[str]:
    """系统里能显示中文的字体（Windows、macOS、常见 Linux 发行版）"""
    win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    paths = [os.path.join(win, n) for n in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "malgun.ttf", "arial.ttf")]
    paths += [
        "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    return paths


def _font(size: int):
    from PIL import ImageFont

    if size not in _FONTS:
        f = None
        for p in _font_paths():
            if os.path.exists(p):
                try:
                    f = ImageFont.truetype(p, size)
                    break
                except OSError:
                    continue
        if f is None:
            try:
                f = ImageFont.load_default(size)
            except TypeError:
                f = ImageFont.load_default()
        _FONTS[size] = f
    return _FONTS[size]


def draw(req: dict) -> None:
    """批量识别的视频截图：像标注页一样画框（分割画轮廓），框的左上角写类别和分数；没有框的（分类）写在左上角"""
    from PIL import Image, ImageDraw

    im = Image.open(req["src"]).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 400))
    font = _font(max(14, round(max(W, H) / 48)))
    next_y = 6
    for it in req.get("items") or []:
        color = it.get("color") or "#e5534b"
        text = f"{it['label']} {float(it.get('score', 0)):.2f}"
        box, poly = it.get("box"), it.get("poly")
        if poly and len(poly) >= 6:
            pts = list(zip(poly[0::2], poly[1::2]))
            d.line(pts + [pts[0]], fill=color, width=lw)
        elif box:
            x, y, w, h = box
            d.rectangle([x, y, x + w, y + h], outline=color, width=lw)
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        pad = max(3, th // 4)
        bw, bh = tw + 2 * pad, th + 2 * pad
        if box:
            x0, y0 = box[0], box[1] - bh
            if y0 < 0:  # 框贴着上边，标签放到框里面
                y0 = box[1]
        else:
            x0, y0 = 6, next_y
            next_y += bh + 4
        x0 = min(max(0, x0), max(0, W - bw))
        d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=color)
        d.text((x0 + pad - l, y0 + pad - t), text, fill="#ffffff", font=font)
    im.save(req["dst"], "JPEG", quality=92)
    emit("done")


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
            elif cmd == "draw":
                draw(req)
            elif cmd == "quit":
                break
            else:
                emit("error", msg=tr("未知命令：{cmd}", cmd=cmd))
        except Exception as e:
            emit("error", msg=str(e))
