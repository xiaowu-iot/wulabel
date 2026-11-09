"""标注辅助模型（第一次用到时才加载，需要 transformers）：

- SAM 交互分割：点 / 框提示 → 目标掩膜 → 多边形
  有 NVIDIA 显卡用 facebook/sam-vit-base（Apache-2.0），否则用小得多的 Zigeng/SlimSAM-uniform-77（Apache-2.0）
- 开放词汇检测（文本提示）：英文类别名 → 目标框
  有 NVIDIA 显卡用 google/owlv2-base-patch16-ensemble，否则用更快的 google/owlvit-base-patch32（都是 Apache-2.0）

模型从 Hugging Face 下载（桌面端会按下载源设置 HF_ENDPOINT 国内镜像），缓存在训练环境目录。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import load_image
from .protocol import log


def _transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError("缺少 transformers 组件，请重试（软件会自动安装）") from e
    import transformers

    return transformers


def mask_to_poly(mask, thresh: float = 0.5):
    """一个实例的概率掩膜（或 0/1 掩膜）转成多边形点 [x1, y1, x2, y2, ...]；失败返回 None。需要 opencv。"""
    try:
        import cv2
    except Exception:
        return None
    m = (np.asarray(mask) > thresh).astype("uint8")
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


def _mask_box(mask) -> list[float]:
    ys, xs = np.nonzero(np.asarray(mask) > 0.5)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1, y0, y1 = float(xs.min()), float(xs.max() + 1), float(ys.min()), float(ys.max() + 1)
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


class Sam:
    def __init__(self, device: torch.device):
        tf = _transformers()
        name = "facebook/sam-vit-base" if device.type == "cuda" else "Zigeng/SlimSAM-uniform-77"
        log(f"加载 SAM 模型 {name}（第一次会下载）…")
        self.processor = tf.SamProcessor.from_pretrained(name)
        self.model = tf.SamModel.from_pretrained(name).to(device).eval()
        self.device = device
        self.key = None
        log("SAM 已就绪")

    @torch.no_grad()
    def _embed(self, path: str):
        """同一张图只算一次图像特征，后面每次点击只跑很快的解码部分"""
        p = Path(path)
        key = (str(p), p.stat().st_mtime)
        if key == self.key:
            return
        img = load_image(p)
        inputs = self.processor(images=img, return_tensors="pt")
        self.orig = inputs["original_sizes"]
        self.reshaped = inputs["reshaped_input_sizes"]
        self.emb = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        # 提示坐标要按图片缩放比例换算到模型输入尺寸
        self.scale = float(self.reshaped[0][0]) / float(self.orig[0][0])
        self.key = key

    def _masks(self, out, multimask: bool):
        masks = self.processor.image_processor.post_process_masks(out.pred_masks.cpu(), self.orig, self.reshaped)[0]
        return masks.numpy(), out.iou_scores.cpu().numpy()[0]

    @torch.no_grad()
    def segment(self, path: str, points: list, box: list | None) -> dict:
        """points: [[x, y, 1 包含 / 0 排除], ...]，box: [x0, y0, x1, y1]，原图像素坐标"""
        self._embed(path)
        s = self.scale
        kw = {}
        if points:
            kw["input_points"] = torch.tensor([[[[p[0] * s, p[1] * s] for p in points]]], dtype=torch.float32, device=self.device)
            kw["input_labels"] = torch.tensor([[[int(p[2]) for p in points]]], dtype=torch.long, device=self.device)
        if box:
            kw["input_boxes"] = torch.tensor([[[v * s for v in box]]], dtype=torch.float32, device=self.device)
        # 只有一个点时有歧义（点在衣服上：要衣服还是整个人？），出 3 个候选取最好的
        multi = len(points) <= 1 and not box
        out = self.model(image_embeddings=self.emb, multimask_output=multi, **kw)
        masks, scores = self._masks(out, multi)  # masks [1, k, H, W]，scores [1, k]
        k = int(np.argmax(scores[0]))
        m = masks[0, k]
        poly = mask_to_poly(m.astype(np.float32)) or []
        return {"points": poly, "box": _mask_box(m), "score": round(float(scores[0, k]), 4)}

    @torch.no_grad()
    def segment_boxes(self, path: str, boxes: list) -> list:
        """多个框各出一个轮廓（文本提示 / 巡瞳 AI 的框转成分割轮廓用）。boxes: [[x0, y0, x1, y1], ...]"""
        if not boxes:
            return []
        self._embed(path)
        s = self.scale
        res = []
        # 分批，避免框太多时显存不够
        for i in range(0, len(boxes), 16):
            chunk = boxes[i : i + 16]
            ib = torch.tensor([[[v * s for v in b] for b in chunk]], dtype=torch.float32, device=self.device)
            out = self.model(image_embeddings=self.emb, input_boxes=ib, multimask_output=False)
            masks, scores = self._masks(out, False)  # [n, 1, H, W]
            for j in range(len(chunk)):
                poly = mask_to_poly(masks[j, 0].astype(np.float32)) or []
                res.append({"points": poly, "score": round(float(scores[j, 0]), 4)})
        return res


class Owl:
    def __init__(self, device: torch.device):
        tf = _transformers()
        self.v2 = device.type == "cuda"
        if self.v2:
            name = "google/owlv2-base-patch16-ensemble"
            self.processor = tf.Owlv2Processor.from_pretrained(name)
            model = tf.Owlv2ForObjectDetection.from_pretrained(name)
        else:
            name = "google/owlvit-base-patch32"
            self.processor = tf.OwlViTProcessor.from_pretrained(name)
            model = tf.OwlViTForObjectDetection.from_pretrained(name)
        log(f"已加载文本检测模型 {name}")
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def detect(self, path: str, queries: list[str], conf: float) -> list:
        """返回 [(x0, y0, x1, y1), 查询序号, 分数]，原图像素坐标，已做 NMS"""
        from torchvision.ops import batched_nms

        img = load_image(Path(path))
        w, h = img.size
        inputs = self.processor(text=[queries], images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        # OWLv2 会把图片补成正方形，换算坐标要用补齐后的边长
        side = max(w, h)
        target = torch.tensor([[side, side]] if self.v2 else [[h, w]], device=self.device)
        r = self.processor.post_process_object_detection(outputs=out, threshold=conf, target_sizes=target)[0]
        boxes, scores, labels = r["boxes"].float(), r["scores"].float(), r["labels"]
        if len(boxes) == 0:
            return []
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        keep = batched_nms(boxes, scores, labels, 0.5)
        res = []
        for i in keep.tolist():
            x0, y0, x1, y1 = boxes[i].tolist()
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                res.append(((x0, y0, x1, y1), int(labels[i]), float(scores[i])))
        return res
