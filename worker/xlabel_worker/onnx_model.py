"""外部 ONNX 检测模型：别处训好的 YOLO 等模型导进来，用于预标注、体检、评估、误报测试。

支持的输出格式（kind）：
    yolov8   YOLOv8 / v11（Ultralytics）：输出 [1, 4+C, N]，cx cy w h + 各类分数
    yolov5   YOLOv5 / v7：输出 [1, N, 5+C]，cx cy w h + 目标分数 + 各类分数
    yolo_e2e YOLOv10 等端到端：输出 [1, N, 6]，x1 y1 x2 y2 分数 类别（已做过 NMS）
    yolox    YOLOX 官方导出：输出 [1, N, 5+C]；BGR、0~255 不归一化、左上角补边，没解码的原始输出会自动解码
    rtdetr   RT-DETR（Ultralytics）：输出 [1, N, 4+C]，归一化 cx cy w h + 各类分数
    detr     DETR / RF-DETR：两个输出，框 [1, N, 4]（归一化 cx cy w h）+ 分数 [1, N, C 或 C+1]
    wulabel  Wulabel 自己导出的：输出 boxes / labels / scores
所有坐标最后都换回原图像素。只需要 onnxruntime 和 numpy；NMS 用 torchvision。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image

KINDS = ["yolov8", "yolov5", "yolo_e2e", "yolox", "rtdetr", "detr", "wulabel"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _session(path: str):
    import onnxruntime as ort

    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in avail] + ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def _dim(v, default: int = 0) -> int:
    return v if isinstance(v, int) and v > 0 else default


def _parse_names(meta: dict) -> list[str]:
    """Ultralytics 导出的模型在元数据里带类别名：names = "{0: 'person', 1: 'car'}" """
    raw = meta.get("names")
    if not raw:
        return []
    try:
        v = ast.literal_eval(raw)
    except Exception:
        try:
            v = json.loads(raw)
        except Exception:
            return []
    if isinstance(v, dict):
        return [str(v[k]) for k in sorted(v, key=lambda x: int(x))]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return []


def inspect(path: str) -> dict:
    """读模型的输入输出和元数据，猜格式、类别名、类别数、输入尺寸"""
    sess = _session(path)
    meta = sess.get_modelmeta().custom_metadata_map or {}
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    in_shape = list(ins[0].shape)
    size = _dim(in_shape[-1], 0) or _dim(in_shape[-2], 0)
    if not size and meta.get("imgsz"):
        try:
            size = int(ast.literal_eval(meta["imgsz"])[0])
        except Exception:
            pass
    size = size or 640
    names = _parse_names(meta)
    desc = (meta.get("description") or "") + " " + (meta.get("model") or "")
    ultra = "names" in meta or "Ultralytics" in desc
    out_names = {o.name for o in outs}
    kind, nc, note = "yolov8", len(names), ""
    if {"boxes", "labels", "scores"} <= out_names:
        kind, note = "wulabel", "Wulabel 导出的模型"
    elif len(outs) >= 2:
        kind = "detr"
        shapes = [list(o.shape) for o in outs]
        sc = next((s for s in shapes if _dim(s[-1]) != 4), shapes[-1])
        nc = nc or _dim(sc[-1])
    else:
        s = list(outs[0].shape)
        a, b = _dim(s[1] if len(s) > 2 else 0), _dim(s[-1])
        if "RT-DETR" in desc or "RTDETR" in desc.replace("-", "").upper():
            kind, nc = "rtdetr", nc or max(b - 4, 0)
        elif b == 6 and a and a <= 1000:
            kind, note = "yolo_e2e", "端到端输出（模型里已做过 NMS）"
        elif a and b and a < b:
            kind, nc = "yolov8", nc or a - 4
        elif "YOLOv5" in desc or ultra:
            kind, nc = "yolov5", nc or max(b - 5, 0)
        elif a and b:
            # 没有元数据的 [N, 5+C] 多半是 YOLOX 官方导出
            kind, nc = "yolox", nc or max(b - 5, 0)
            note = "没有类别名，请手动填写"
    if not names and nc:
        names = [f"class_{i}" for i in range(nc)]
    return {
        "kind": kind,
        "classes": names,
        "numClasses": nc,
        "imgSize": size,
        "inputs": [{"name": i.name, "shape": [str(d) for d in i.shape]} for i in ins],
        "outputs": [{"name": o.name, "shape": [str(d) for d in o.shape]} for o in outs],
        "note": note,
    }


def _nms(boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, iou: float = 0.5):
    import torch
    from torchvision.ops import batched_nms

    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    keep = batched_nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), torch.from_numpy(cls).long(), iou)
    return keep.numpy()


class OnnxDetector:
    """info：kind、classes、imgSize（导入时写进 model.json）"""

    def __init__(self, path: str, info: dict):
        self.sess = _session(path)
        self.kind = info.get("kind", "yolov8")
        self.classes = list(info.get("classes") or [])
        self.size = int(info.get("imgSize") or 640)
        self.input = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        # Wulabel 动态尺寸模型的输入是 [3, H, W]，没有 batch 维
        self.no_batch = self.kind == "wulabel" and len(shape) == 3

    # ---------- 预处理 ----------

    def _letterbox(self, img: Image.Image, center: bool):
        w, h = img.size
        s = self.size
        r = min(s / w, s / h)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        canvas = Image.new("RGB", (s, s), (114, 114, 114))
        px, py = ((s - nw) // 2, (s - nh) // 2) if center else (0, 0)
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        return np.asarray(canvas, dtype=np.float32), r, px, py

    def _run(self, x: np.ndarray):
        return self.sess.run(None, {self.input: x})

    # ---------- 推理 ----------

    def predict(self, img: Image.Image, conf: float) -> list:
        """返回 [(x0, y0, x1, y1, 类别序号, 分数)]，原图像素坐标"""
        w, h = img.size
        k = self.kind
        if k == "wulabel":
            if self.no_batch:
                # 动态尺寸：原图直接进
                x, r = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0, 1.0
            else:
                # 固定尺寸：按比例缩放后在右下补边到 S×S，输出坐标再按比例换回
                arr, r, _, _ = self._letterbox(img, center=False)
                x = (arr.transpose(2, 0, 1) / 255.0)[None]
            boxes, labels, scores = self._run(np.ascontiguousarray(x, dtype=np.float32))[:3]
            boxes, labels, scores = np.asarray(boxes).reshape(-1, 4), np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
            boxes = boxes / r
            m = scores >= conf
            return [(*b, int(l) - 1, float(sc)) for b, l, sc in zip(boxes[m].tolist(), labels[m], scores[m])]

        if k in ("rtdetr", "detr"):
            s = self.size
            arr = np.asarray(img.resize((s, s), Image.BILINEAR), dtype=np.float32) / 255.0
            if k == "detr":
                arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            outs = self._run(arr.transpose(2, 0, 1)[None].astype(np.float32))
            if k == "rtdetr":
                o = np.asarray(outs[0])[0]
                bx, sc_all = o[:, :4], o[:, 4:]
            else:
                a, b = np.asarray(outs[0])[0], np.asarray(outs[1])[0]
                bx, logits = (a, b) if a.shape[-1] == 4 else (b, a)
                if len(self.classes) and logits.shape[-1] == len(self.classes) + 1:
                    e = np.exp(logits - logits.max(-1, keepdims=True))
                    sc_all = (e / e.sum(-1, keepdims=True))[:, :-1]  # 去掉最后的「无目标」类
                else:
                    sc_all = 1 / (1 + np.exp(-logits))
            cls = sc_all.argmax(1)
            score = sc_all.max(1)
            m = score >= conf
            cx, cy, bw, bh = bx[m, 0], bx[m, 1], bx[m, 2], bx[m, 3]
            if bx.max() <= 1.5:  # 归一化坐标
                cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            else:
                cx, cy, bw, bh = cx * w / s, cy * h / s, bw * w / s, bh * h / s
            out = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
            return self._finish(out, score[m], cls[m], w, h, nms=False)

        # YOLO 系
        yolox = k == "yolox"
        arr, r, px, py = self._letterbox(img, center=not yolox)
        if yolox:
            x = arr[:, :, ::-1].transpose(2, 0, 1)  # BGR、0~255
        else:
            x = arr.transpose(2, 0, 1) / 255.0
        o = np.asarray(self._run(np.ascontiguousarray(x[None], dtype=np.float32))[0])[0]
        if k == "yolo_e2e":
            boxes, score, cls = o[:, :4], o[:, 4], o[:, 5].astype(int)
            m = score >= conf
            boxes = boxes[m].copy()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
            return self._finish(boxes, score[m], cls[m], w, h, nms=False)
        if k == "yolov8":
            if o.shape[0] < o.shape[1]:
                o = o.T  # [4+C, N] → [N, 4+C]
            xywh, sc_all = o[:, :4], o[:, 4:]
        else:
            if yolox:
                o = self._yolox_decode(o)
            xywh, sc_all = o[:, :4], o[:, 5:] * o[:, 4:5]
        cls = sc_all.argmax(1)
        score = sc_all.max(1)
        m = score >= conf
        xywh, score, cls = xywh[m], score[m], cls[m]
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
        return self._finish(boxes, score, cls, w, h, nms=True)

    def _yolox_decode(self, o: np.ndarray) -> np.ndarray:
        """YOLOX 官方导出默认不在模型里解码：输出是相对网格的偏移，这里按步长 8/16/32 解码。已经解码过的原样返回"""
        s = self.size
        grids, strides = [], []
        for st in (8, 16, 32):
            n = s // st
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack([xv, yv], -1).reshape(-1, 2))
            strides.append(np.full((n * n, 1), st))
        grid = np.concatenate(grids).astype(np.float32)
        stride = np.concatenate(strides).astype(np.float32)
        if len(grid) != len(o) or np.abs(o[:, :2]).max() > 5:
            return o
        o = o.copy()
        o[:, :2] = (o[:, :2] + grid) * stride
        o[:, 2:4] = np.exp(o[:, 2:4]) * stride
        return o

    def _finish(self, boxes, score, cls, w, h, nms: bool) -> list:
        if len(boxes) == 0:
            return []
        boxes = boxes.astype(np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        ok = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        boxes, score, cls = boxes[ok], np.asarray(score)[ok], np.asarray(cls)[ok]
        idx = _nms(boxes, score, cls) if nms else np.arange(len(boxes))
        idx = idx[np.argsort(-score[idx])][:300]
        return [(*boxes[i].tolist(), int(cls[i]), float(score[i])) for i in idx]


def load(path: Path) -> tuple[OnnxDetector, dict]:
    """加载导入的外部模型：path 是 model.onnx，旁边的 model.json 记着格式和类别"""
    info = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    return OnnxDetector(str(path), info), info
