"""检测指标：mAP（COCO 101 点插值）、精确率/召回率、混淆矩阵、错例统计。只依赖 numpy。

约定：
- 框格式 xyxy，像素坐标
- 类别序号从 0 开始
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


@dataclass
class ImagePred:
    boxes: np.ndarray  # (N, 4)
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,)


@dataclass
class ImageGt:
    boxes: np.ndarray  # (M, 4)
    labels: np.ndarray  # (M,)
    image_id: int = 0


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """101 点插值 AP，与 COCO 的算法一致"""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    # 精确率做成单调不增
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def _match(pred: ImagePred, gt: ImageGt, cls: int, thr: float):
    """单张图、单个类别、单个阈值的贪心匹配，返回该类预测的 tp 标记和分数"""
    pm = pred.labels == cls
    gm = gt.labels == cls
    pb, ps = pred.boxes[pm], pred.scores[pm]
    gb = gt.boxes[gm]
    order = np.argsort(-ps, kind="stable")
    pb, ps = pb[order], ps[order]
    tp = np.zeros(len(pb), dtype=bool)
    if len(gb):
        ious = box_iou(pb, gb)
        used = np.zeros(len(gb), dtype=bool)
        for i in range(len(pb)):
            cand = np.where(~used & (ious[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                used[j] = True
                tp[i] = True
    return tp, ps, int(gm.sum())


@dataclass
class EvalResult:
    map50: float
    map: float
    per_class: list[dict] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    confusion: list[list[int]] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


def evaluate(preds: list[ImagePred], gts: list[ImageGt], num_classes: int, conf: float = 0.25) -> EvalResult:
    ap = np.full((num_classes, len(IOU_THRESHOLDS)), np.nan)
    per_class = []
    total_tp = total_fp = total_gt = 0
    for c in range(num_classes):
        n_gt_c = 0
        for t, thr in enumerate(IOU_THRESHOLDS):
            tps, scs = [], []
            n_gt = 0
            for p, g in zip(preds, gts):
                tp, s, n = _match(p, g, c, thr)
                tps.append(tp)
                scs.append(s)
                n_gt += n
            tp_all = np.concatenate(tps) if tps else np.zeros(0, bool)
            sc_all = np.concatenate(scs) if scs else np.zeros(0)
            ap[c, t] = average_precision(tp_all, sc_all, n_gt)
            if t == 0:
                n_gt_c = n_gt
                keep = sc_all >= conf
                ctp = int(tp_all[keep].sum())
                cfp = int((~tp_all[keep]).sum())
                total_tp += ctp
                total_fp += cfp
                total_gt += n_gt
                per_class.append(
                    {
                        "index": c,
                        "gt": n_gt,
                        "precision": ctp / max(ctp + cfp, 1),
                        "recall": ctp / max(n_gt, 1),
                    }
                )
        per_class[c]["ap50"] = None if np.isnan(ap[c, 0]) else float(ap[c, 0])
        per_class[c]["ap"] = None if np.all(np.isnan(ap[c])) else float(np.nanmean(ap[c]))
        per_class[c]["gt"] = n_gt_c

    valid = ~np.isnan(ap[:, 0])
    map50 = float(np.mean(ap[valid, 0])) if valid.any() else 0.0
    map_all = float(np.nanmean(ap[valid])) if valid.any() else 0.0
    cm, errors = confusion_and_errors(preds, gts, num_classes, conf)
    return EvalResult(
        map50=map50,
        map=map_all,
        per_class=per_class,
        precision=total_tp / max(total_tp + total_fp, 1),
        recall=total_tp / max(total_gt, 1),
        confusion=cm,
        errors=errors,
    )


def confusion_and_errors(preds, gts, num_classes: int, conf: float, iou: float = 0.5):
    """混淆矩阵（最后一行/列是背景）和每张图的错误统计。行是真实类别，列是预测类别。"""
    n = num_classes
    cm = np.zeros((n + 1, n + 1), dtype=int)
    errors = []
    for p, g in zip(preds, gts):
        keep = p.scores >= conf
        pb, pl = p.boxes[keep], p.labels[keep]
        ious = box_iou(g.boxes, pb)
        matched_p = set()
        missed = wrong = 0
        # 每个真实框找重叠最大的预测框（不区分类别）
        pairs = sorted(
            ((ious[i, j], i, j) for i in range(len(g.boxes)) for j in range(len(pb)) if ious[i, j] >= iou),
            reverse=True,
        )
        matched_g = set()
        for _, i, j in pairs:
            if i in matched_g or j in matched_p:
                continue
            matched_g.add(i)
            matched_p.add(j)
            cm[int(g.labels[i]), int(pl[j])] += 1
            if int(g.labels[i]) != int(pl[j]):
                wrong += 1
        for i in range(len(g.boxes)):
            if i not in matched_g:
                cm[int(g.labels[i]), n] += 1
                missed += 1
        extra = 0
        for j in range(len(pb)):
            if j not in matched_p:
                cm[n, int(pl[j])] += 1
                extra += 1
        if missed or extra or wrong:
            errors.append({"imageId": int(g.image_id), "missed": missed, "extra": extra, "wrong": wrong})
    errors.sort(key=lambda e: -(e["missed"] + e["extra"] + e["wrong"]))
    return cm.tolist(), errors
