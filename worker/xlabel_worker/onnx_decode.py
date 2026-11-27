"""通用 ONNX 模型解码：检测、实例分割、关键点、分类，Wulabel 自己导出的和 Ultralytics 等常见格式都认。

这个文件不依赖 xlabel_worker 里的其他模块，也会原样复制进推理服务包（只需要 onnxruntime、numpy、opencv）。

格式（kind）：
    wulabel   Wulabel 导出：boxes / labels（从 1 开始）/ scores [/ masks | keypoints, keypoints_scores]，分类是 probs
    yolov8    Ultralytics 原始输出（YOLOv5u / v8 / v9 / 11 / 12 等）：
                检测 [1, 4+C, N]；分割 [1, 4+C+32, N] + 原型 [1, 32, H/4, W/4]；
                关键点 [1, 4+C+K*3, N]；分类 [1, C]（已经过 softmax）
    yolo_e2e  端到端（YOLOv10 / YOLO26 等，模型里已去重）：[1, N, 6(+掩膜系数 | +关键点)]，x1 y1 x2 y2 分数 类别
    yolov5    YOLOv5 / v7：[1, N, 5+C]
    yolox     YOLOX 官方导出：[1, N, 5+C]，BGR、0~255、左上角补边，没解码的会自动解码
    rtdetr    RT-DETR：[1, N, 4+C]，归一化 cx cy w h
    detr      DETR / RF-DETR：框 [1, N, 4] + 分数 [1, N, C 或 C+1]

predict() 的结果每个目标一项：
    {"cls": 类别序号, "score": 分数, "box": [x0, y0, x1, y1]（原图像素）,
     "polygon": [x1, y1, ...]（分割）, "keypoints": [x, y, v, ...]（关键点，v：2 找到、0 没找到）}
    分类：{"cls", "score"}，没有 box，按分数从高到低
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import cv2
import numpy as np

KINDS = ["wulabel", "yolov8", "yolo_e2e", "yolov5", "yolox", "rtdetr", "detr"]
TASKS = ["detect", "segment", "pose", "classify"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------- 读模型信息 ----------------


def session(path, use_gpu: bool = True):
    import onnxruntime as ort

    avail = ort.get_available_providers()
    wanted = ("CUDAExecutionProvider", "CoreMLExecutionProvider") if use_gpu else ()
    providers = [p for p in wanted if p in avail] + ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def _dim(v, default: int = 0) -> int:
    return v if isinstance(v, int) and v > 0 else default


def _literal(raw):
    if raw is None or raw == "":
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return None


def parse_names(raw) -> list[str]:
    """Ultralytics 在元数据里写类别名：names = "{0: 'person', 1: 'car'}" """
    v = _literal(raw)
    if isinstance(v, dict):
        return [str(v[k]) for k in sorted(v, key=lambda x: int(x))]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return []


def inspect_session(sess) -> dict:
    """看输入输出和元数据，判断格式、任务、类别、输入尺寸。note 是提示代码（由调用方翻译）"""
    meta = sess.get_modelmeta().custom_metadata_map or {}
    ins, outs = sess.get_inputs(), sess.get_outputs()
    in_shape = list(ins[0].shape)
    size = _dim(in_shape[-1]) or _dim(in_shape[-2])
    if not size:
        v = _literal(meta.get("imgsz"))
        if isinstance(v, (list, tuple)) and v:
            size = int(v[0])
        elif isinstance(v, int):
            size = v
    size = size or 640
    names = parse_names(meta.get("names"))
    kpt = _literal(meta.get("kpt_shape"))
    kpt_shape = [int(kpt[0]), int(kpt[1])] if isinstance(kpt, (list, tuple)) and len(kpt) == 2 else None
    desc = f"{meta.get('description', '')} {meta.get('model', '')}"
    ultra = "names" in meta or "Ultralytics" in desc
    meta_task = str(meta.get("task") or "").lower()
    out_names = {o.name for o in outs}
    shapes = [list(o.shape) for o in outs]
    kind, task, nc, note = "yolov8", "detect", len(names), ""

    if "probs" in out_names:
        kind, task, note = "wulabel", "classify", "wulabel"
    elif {"boxes", "labels", "scores"} <= out_names:
        kind, note = "wulabel", "wulabel"
        task = "segment" if "masks" in out_names else "pose" if "keypoints" in out_names else "detect"
    elif meta_task == "classify" or (len(outs) == 1 and len(shapes[0]) == 2):
        kind, task = "yolov8", "classify"
        nc = nc or _dim(shapes[0][-1])
    elif len(outs) == 2 and len(shapes[1]) == 4:
        # 分割：检测输出 + 掩膜原型
        task = "segment"
        s = shapes[0]
        a, b = _dim(s[1]), _dim(s[2])
        nm = _dim(shapes[1][1], 32)
        if nc and a == 4 + nc + nm:
            kind = "yolov8"
        elif b == 6 + nm or (not nc and a and b and a > b):
            kind, note = "yolo_e2e", "e2e"
        else:
            kind = "yolov8"
            nc = nc or max(a - 4 - nm, 0)
    elif len(outs) >= 2:
        kind = "detr"
        sc = next((s for s in shapes if _dim(s[-1]) != 4), shapes[-1])
        nc = nc or _dim(sc[-1])
    else:
        s = shapes[0]
        a, b = _dim(s[1] if len(s) > 2 else 0), _dim(s[-1])
        if meta_task == "pose" or kpt_shape:
            task = "pose"
            nk = (kpt_shape[0] * kpt_shape[1]) if kpt_shape else 51
            if (nc or 1) and a == 4 + (nc or 1) + nk:
                kind = "yolov8"
            elif b == 6 + nk or (a and b and a > b):
                kind, note = "yolo_e2e", "e2e"
            else:
                kind = "yolov8"
        elif "RT-DETR" in desc or "RTDETR" in desc.replace("-", "").upper():
            kind, nc = "rtdetr", nc or max(b - 4, 0)
        elif nc and a == 4 + nc:
            kind = "yolov8"
        elif b == 6 and a and a <= 1000:
            kind, note = "yolo_e2e", "e2e"
        elif a and b and a < b:
            kind, nc = "yolov8", nc or a - 4
        elif "YOLOv5" in desc or ultra:
            kind, nc = "yolov5", nc or max(b - 5, 0)
        elif a and b:
            # 没有元数据的 [N, 5+C] 多半是 YOLOX 官方导出
            kind, nc, note = "yolox", nc or max(b - 5, 0), "no_names"
    if task == "pose" and not kpt_shape:
        kpt_shape = [17, 3]
    if task == "pose" and not nc:
        nc = 1
    if not names and nc:
        names = [f"class_{i}" for i in range(nc)]
    return {
        "kind": kind,
        "task": task,
        "classes": names,
        "numClasses": nc or len(names),
        "imgSize": int(size),
        "kptShape": kpt_shape,
        "inputs": [{"name": i.name, "shape": [str(d) for d in i.shape]} for i in ins],
        "outputs": [{"name": o.name, "shape": [str(d) for d in o.shape]} for o in outs],
        "note": note,
    }


# ---------------- 通用工具 ----------------


def nms(boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, iou: float = 0.5, top: int = 300) -> np.ndarray:
    """按类别分别去重（同类重叠超过 iou 的只留分数最高的），返回保留的下标（分数从高到低）"""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    off = boxes + (cls.astype(np.float32) * (float(boxes.max()) + 1))[:, None]
    x0, y0, x1, y1 = off[:, 0], off[:, 1], off[:, 2], off[:, 3]
    area = (x1 - x0).clip(0) * (y1 - y0).clip(0)
    order = np.argsort(-scores)
    keep = []
    while len(order) and len(keep) < top:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = (np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest])).clip(0)
        h = (np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest])).clip(0)
        inter = w * h
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= iou]
    return np.array(keep, dtype=int)


def mask_to_poly(mask: np.ndarray, thresh: float = 0.5, min_area: float = 4) -> list[float]:
    """概率掩膜（或 0/1）转成多边形 [x1, y1, ...]，取面积最大的一块；失败返回 []"""
    m = (np.asarray(mask) > thresh).astype(np.uint8)
    if int(m.sum()) < min_area:
        return []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return []
    pts = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(pts) < 3:
        raw = c.reshape(-1, 2)
        pts = raw[:: max(1, len(raw) // 24)]
        if len(pts) < 3:
            return []
    return [float(v) for p in pts for v in p]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------- 模型 ----------------


class OnnxModel:
    """info：model.json 的内容（kind、task、classes、imgSize、kptShape、multiLabel），缺的项自动从模型里读"""

    def __init__(self, path, info: dict | None = None, use_gpu: bool = True, max_side: int | None = None):
        self.sess = session(path, use_gpu)
        auto = inspect_session(self.sess)
        info = dict(info or {})
        self.kind = info.get("kind") or auto["kind"]
        self.task = info.get("task") or (auto["task"] if self.kind == auto["kind"] else "detect")
        self.classes = list(info.get("classes") or auto["classes"])
        self.size = int(info.get("imgSize") or auto["imgSize"])
        kpt = info.get("kptShape") or auto.get("kptShape")
        self.kpt_shape = [int(kpt[0]), int(kpt[1])] if kpt else None
        self.multi = bool(info.get("multiLabel"))
        self.max_side = max_side
        inp = self.sess.get_inputs()[0]
        self.input = inp.name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        # Wulabel 动态尺寸模型的输入是 [3, H, W]，没有 batch 维
        self.no_batch = self.kind == "wulabel" and len(inp.shape) == 3
        if self.task == "classify" and _dim(inp.shape[-1]):
            self.size = int(inp.shape[-1])

    @property
    def provider(self) -> str:
        return self.sess.get_providers()[0]

    def _run(self, x: np.ndarray) -> list:
        return self.sess.run(None, {self.input: np.ascontiguousarray(x, dtype=np.float32)})

    def _letterbox(self, rgb: np.ndarray, center: bool):
        h, w = rgb.shape[:2]
        s = self.size
        r = min(s / w, s / h)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        px, py = ((s - nw) // 2, (s - nh) // 2) if center else (0, 0)
        canvas[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas.astype(np.float32), r, px, py

    # ---------- 入口 ----------

    def predict(self, rgb: np.ndarray, conf: float = 0.25) -> list[dict]:
        """rgb：H×W×3 的 uint8 数组（RGB 顺序）"""
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, -1)
        rgb = rgb[:, :, :3]
        if self.task == "classify":
            return self._classify(rgb, conf)
        if self.kind == "wulabel":
            return self._wulabel(rgb, conf)
        if self.kind in ("rtdetr", "detr"):
            return self._detr(rgb, conf)
        return self._yolo(rgb, conf)

    # ---------- 分类 ----------

    def _classify(self, rgb: np.ndarray, conf: float) -> list[dict]:
        s = self.size
        if self.kind == "wulabel":
            # Wulabel 导出的分类模型里已经做了归一化，输入直接缩放到 S×S、0~1
            img = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA)
        else:
            # Ultralytics：短边缩放到 S 再从中间裁 S×S
            h, w = rgb.shape[:2]
            r = s / min(h, w)
            img = cv2.resize(rgb, (max(s, round(w * r)), max(s, round(h * r))), interpolation=cv2.INTER_AREA)
            y0, x0 = (img.shape[0] - s) // 2, (img.shape[1] - s) // 2
            img = img[y0:y0 + s, x0:x0 + s]
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        probs = np.asarray(self._run(x)[0]).reshape(-1)
        if not self.multi and (probs.min() < 0 or abs(float(probs.sum()) - 1) > 0.05):
            e = np.exp(probs - probs.max())
            probs = e / e.sum()
        order = np.argsort(-probs)
        if not self.multi:
            order = order[:5]
        return [{"cls": int(i), "score": float(probs[i])} for i in order if probs[i] >= conf]

    # ---------- Wulabel 格式 ----------

    def _wulabel(self, rgb: np.ndarray, conf: float) -> list[dict]:
        h, w = rgb.shape[:2]
        if self.no_batch:
            # 动态尺寸：原图（太大时按 max_side 缩小）直接进
            r = min(1.0, self.max_side / max(h, w)) if self.max_side else 1.0
            img = cv2.resize(rgb, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv2.INTER_AREA) if r < 1 else rgb
            x = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        else:
            # 固定尺寸：按比例缩放后在右下补边到 S×S
            arr, r, _, _ = self._letterbox(rgb, center=False)
            x = (arr.transpose(2, 0, 1) / 255.0)[None]
        res = dict(zip(self.out_names, self._run(x)))
        boxes = np.asarray(res["boxes"]).reshape(-1, 4) / r
        labels = np.asarray(res["labels"]).reshape(-1)
        scores = np.asarray(res["scores"]).reshape(-1)
        masks = res.get("masks") if self.task == "segment" else None
        kps = res.get("keypoints") if self.task == "pose" else None
        ks = res.get("keypoints_scores")
        out = []
        for i in range(len(scores)):
            if scores[i] < conf:
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[i]]
            d = {"cls": int(labels[i]) - 1, "score": float(scores[i]), "box": [max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)]}
            if masks is not None:
                m = np.asarray(masks[i]).reshape(np.asarray(masks[i]).shape[-2:])
                d["polygon"] = [v / r for v in mask_to_poly(m)]
            if kps is not None:
                flat = []
                for j, (kx, ky, _) in enumerate(np.asarray(kps[i])):
                    found = ks is None or np.asarray(ks)[i, j] > 0
                    flat += [float(kx) / r, float(ky) / r, 2] if found else [0.0, 0.0, 0]
                d["keypoints"] = flat
            out.append(d)
        return out

    # ---------- DETR 系 ----------

    def _detr(self, rgb: np.ndarray, conf: float) -> list[dict]:
        h, w = rgb.shape[:2]
        s = self.size
        arr = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        if self.kind == "detr":
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        outs = self._run(arr.transpose(2, 0, 1)[None])
        if self.kind == "rtdetr":
            o = np.asarray(outs[0])[0]
            bx, sc_all = o[:, :4], o[:, 4:]
        else:
            a, b = np.asarray(outs[0])[0], np.asarray(outs[1])[0]
            bx, logits = (a, b) if a.shape[-1] == 4 else (b, a)
            if len(self.classes) and logits.shape[-1] == len(self.classes) + 1:
                e = np.exp(logits - logits.max(-1, keepdims=True))
                sc_all = (e / e.sum(-1, keepdims=True))[:, :-1]  # 去掉最后的「无目标」类
            else:
                sc_all = _sigmoid(logits)
        cls = sc_all.argmax(1)
        score = sc_all.max(1)
        m = score >= conf
        cx, cy, bw, bh = bx[m, 0], bx[m, 1], bx[m, 2], bx[m, 3]
        if bx.size and bx.max() <= 1.5:  # 归一化坐标
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
        else:
            cx, cy, bw, bh = cx * w / s, cy * h / s, bw * w / s, bh * h / s
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) if len(cx) else np.zeros((0, 4))
        idx = self._clean(boxes, score[m], w, h)
        return [{"cls": int(cls[m][i]), "score": float(score[m][i]), "box": boxes[i].tolist()} for i in idx[np.argsort(-score[m][idx])][:300]]

    # ---------- YOLO 系 ----------

    def _yolo(self, rgb: np.ndarray, conf: float) -> list[dict]:
        h, w = rgb.shape[:2]
        k = self.kind
        yolox = k == "yolox"
        arr, r, px, py = self._letterbox(rgb, center=not yolox)
        x = arr[:, :, ::-1].transpose(2, 0, 1) if yolox else arr.transpose(2, 0, 1) / 255.0  # YOLOX：BGR、0~255
        outs = self._run(x[None])
        o = np.asarray(outs[0])[0]
        proto = np.asarray(outs[1])[0] if (self.task == "segment" and len(outs) > 1) else None
        nm = proto.shape[0] if proto is not None else 0
        nk = self.kpt_shape[0] * self.kpt_shape[1] if (self.task == "pose" and self.kpt_shape) else 0

        extra = None  # 每个候选目标的掩膜系数或关键点
        if k == "yolo_e2e":
            boxes, score, cls = o[:, :4], o[:, 4], o[:, 5].astype(int)
            extra = o[:, 6:6 + (nm or nk)] if (nm or nk) else None
            m = score >= conf
            boxes, score, cls = boxes[m], score[m], cls[m]
            extra = extra[m] if extra is not None else None
            keep = None
        else:
            if k == "yolov8":
                # Ultralytics 原始输出是 [4+C(+extra), N]，换成 [N, 4+C(+extra)]；已经是 [N, …] 的不动
                exp = 4 + len(self.classes) + nm + nk if self.classes else 0
                already = (exp and o.shape[1] == exp and o.shape[0] != exp) or (not exp and o.shape[0] > o.shape[1])
                if not already:
                    o = o.T
                nc = o.shape[1] - 4 - nm - nk
                xywh, sc_all = o[:, :4], o[:, 4:4 + nc]
                if nm or nk:
                    extra = o[:, 4 + nc:]
            else:
                if yolox:
                    o = self._yolox_decode(o)
                xywh, sc_all = o[:, :4], o[:, 5:] * o[:, 4:5]
            cls = sc_all.argmax(1)
            score = sc_all.max(1)
            m = score >= conf
            xywh, score, cls = xywh[m], score[m], cls[m]
            extra = extra[m] if extra is not None else None
            if len(score) > 3000:
                top = np.argsort(-score)[:3000]
                xywh, score, cls = xywh[top], score[top], cls[top]
                extra = extra[top] if extra is not None else None
            boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1) if len(xywh) else np.zeros((0, 4))
            keep = nms(boxes, score, cls)
        if keep is not None:
            boxes, score, cls = boxes[keep], score[keep], cls[keep]
            extra = extra[keep] if extra is not None else None
        else:
            order = np.argsort(-score)[:300]
            boxes, score, cls = boxes[order], score[order], cls[order]
            extra = extra[order] if extra is not None else None

        lb = boxes.copy()  # 模型输入（补边后）坐标，算掩膜用
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - px) / r).clip(0, w)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - py) / r).clip(0, h)
        out = []
        for i in range(len(score)):
            b = boxes[i]
            if b[2] - b[0] < 1 or b[3] - b[1] < 1:
                continue
            d = {"cls": int(cls[i]), "score": float(score[i]), "box": [float(v) for v in b]}
            if proto is not None and extra is not None:
                d["polygon"] = self._mask_poly(extra[i], proto, lb[i], b)
            if nk and extra is not None:
                d["keypoints"] = self._kpts(extra[i], r, px, py, w, h)
            out.append(d)
        return out

    def _mask_poly(self, coef: np.ndarray, proto: np.ndarray, lb: np.ndarray, box: np.ndarray) -> list[float]:
        """掩膜系数 × 原型 → 目标框范围内的掩膜 → 原图上的多边形"""
        nm, mh, mw = proto.shape
        sx, sy = mw / self.size, mh / self.size
        x0, y0 = max(0, int(np.floor(lb[0] * sx))), max(0, int(np.floor(lb[1] * sy)))
        x1, y1 = min(mw, int(np.ceil(lb[2] * sx))), min(mh, int(np.ceil(lb[3] * sy)))
        if x1 <= x0 or y1 <= y0:
            return []
        crop = _sigmoid(coef[:nm] @ proto[:, y0:y1, x0:x1].reshape(nm, -1)).reshape(y1 - y0, x1 - x0)
        bw, bh = max(1, int(round(box[2] - box[0]))), max(1, int(round(box[3] - box[1])))
        up = cv2.resize(crop.astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
        poly = mask_to_poly(up)
        return [v + (float(box[0]) if j % 2 == 0 else float(box[1])) for j, v in enumerate(poly)]

    def _kpts(self, raw: np.ndarray, r: float, px: int, py: int, w: int, h: int) -> list[float]:
        k, dim = self.kpt_shape
        a = np.asarray(raw[: k * dim]).reshape(k, dim)
        flat = []
        for p in a:
            vis = dim < 3 or p[2] > 0.5
            kx, ky = (p[0] - px) / r, (p[1] - py) / r
            if vis and 0 <= kx <= w and 0 <= ky <= h:
                flat += [float(kx), float(ky), 2]
            else:
                flat += [0.0, 0.0, 0]
        return flat

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

    @staticmethod
    def _clean(boxes: np.ndarray, score: np.ndarray, w: int, h: int) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros(0, dtype=int)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return np.where((boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1))[0]


def load(path, use_gpu: bool = True, max_side: int | None = None) -> tuple[OnnxModel, dict]:
    """加载 model.onnx，旁边有 model.json 就用它记的格式和类别"""
    p = Path(path)
    j = p.with_suffix(".json")
    if not j.exists():
        j = p.with_name("model.json")
    info = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
    m = OnnxModel(p, info, use_gpu=use_gpu, max_side=max_side)
    info.setdefault("kind", m.kind)
    info.setdefault("task", m.task)
    info.setdefault("classes", m.classes)
    return m, info
