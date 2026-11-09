"""用现在的数据重新评估一个模型（训练好的 best.pt，或者导入的外部 ONNX 模型）。

配置（JSON）：
    dataset   桌面端刚导出的数据集（分类是分类文件夹，其他是 COCO）
    out       写 eval.json 的目录
    weights   best.pt 或 model.onnx
    classes   项目当前的类别名（数据集里的类别顺序）
模型的类别按名字对应到项目类别，项目里没有的类别不参与评估。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import models
from .data import ClsFolder, CocoDetection, has_cls_split, has_split, load_image
from .metrics import ImageGt, ImagePred, classify_eval, classify_threshold_curve, evaluate, threshold_curve
from .protocol import emit, log
from .train import ClsTask, pick_device, split_report

SPLITS = ("val", "test")


def _remap(model_classes: list[str], classes: list[str]) -> dict[int, int]:
    idx = {n: i for i, n in enumerate(classes)}
    return {mi: idx[n] for mi, n in enumerate(model_classes) if n in idx}


def _det_splits(root: Path, classes: list[str], predict) -> dict:
    """predict(PIL 图) → (boxes xyxy, scores, 项目类别序号)"""
    out = {}
    for split in SPLITS:
        if not has_split(root, split):
            continue
        ds = CocoDetection(root, split, train=False)
        preds, gts = [], []
        for i, im in enumerate(ds.images):
            boxes, scores, labels = predict(load_image(ds.dir / im["file_name"]))
            preds.append(ImagePred(boxes=boxes, scores=scores, labels=labels))
            t = ds.target(i)
            gts.append(ImageGt(boxes=t["boxes"].numpy(), labels=t["labels"].numpy() - 1, image_id=int(im["id"])))
            if i % 20 == 0:
                emit("progress", split=split, done=i + 1, total=len(ds.images))
        r = evaluate(preds, gts, len(classes))
        r.thresholds = threshold_curve(preds, gts, len(classes))
        out[split] = split_report(r, classes, len(ds.images))
        log(f"{'验证集' if split == 'val' else '测试集'}：{len(ds.images)} 张，mAP50 {r.map50:.3f}")
    return out


def run(cfg: dict) -> None:
    root = Path(cfg["dataset"])
    out = Path(cfg["out"])
    weights = Path(cfg["weights"])
    classes: list[str] = cfg["classes"]
    device = pick_device("auto")
    report = {"classes": classes, "bestEpoch": None, "splits": {}, "evaluatedAt": int(time.time()), "reevaluated": True}

    if weights.suffix.lower() == ".onnx":
        from . import onnx_model

        det, info = onnx_model.load(weights)
        cmap = _remap(info.get("classes", []), classes)
        use = info.get("use")
        if not cmap:
            raise RuntimeError("模型的类别名和项目类别一个都对不上，请在导入模型时把类别名改成项目里的名字")
        log(f"外部模型：{len(cmap)} 个类别和项目类别对应上")

        def predict(img):
            dets = [d for d in det.predict(img, 0.05) if d[4] in cmap and (use is None or d[4] in use)]
            if not dets:
                return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
            a = np.array([d[:4] for d in dets], dtype=np.float32)
            return a, np.array([d[5] for d in dets]), np.array([cmap[d[4]] for d in dets], dtype=int)

        report["task"] = "detect"
        report["splits"] = _det_splits(root, classes, predict)
    else:
        model, ck = models.load(weights, device)
        key = ck["model"]
        report["task"] = models.task_of(key)
        cmap = _remap(ck["classes"], classes)
        if models.is_cls(key):
            task = ClsTask({"img_size": ck["img_size"]}, root, key)
            for split in SPLITS:
                if not has_cls_split(root, split):
                    continue
                ds = ClsFolder(root, split, train=False, size=int(ck["img_size"]))
                y_true, y_pred, ids, scores = [], [], [], []
                model.eval()
                with torch.no_grad():
                    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, collate_fn=task.collate)
                    for imgs, labels, idl in loader:
                        prob = torch.softmax(model(imgs.to(device)).float(), dim=1).cpu()
                        conf, pred = prob.max(dim=1)
                        for t, pi, s, i in zip(labels.tolist(), pred.tolist(), conf.tolist(), idl):
                            # 模型的类别对不上项目类别的预测当成分错（-1 不会等于真实类别）
                            y_true.append(t)
                            y_pred.append(cmap.get(pi, -1))
                            scores.append(s)
                            ids.append(i)
                # 分错到项目里没有的类别时，混淆矩阵放不下，记成「判成第一个类以外」——这里归到最后统计前过滤
                yp = [p if p >= 0 else (t + 1) % max(len(classes), 1) for p, t in zip(y_pred, y_true)]
                r = classify_eval(y_true, yp, ids, len(classes), scores=np.array(scores))
                r.thresholds = classify_threshold_curve(y_true, yp, scores)
                report["splits"][split] = split_report(r, classes, len(ds))
                log(f"{'验证集' if split == 'val' else '测试集'}：{len(ds)} 张，准确率 {r.map50:.3f}")
        else:
            from torchvision.transforms import functional as F

            model.eval()

            @torch.no_grad()
            def predict(img):
                o = model([F.to_tensor(img).to(device)])[0]
                b = o["boxes"].float().cpu().numpy()
                s = o["scores"].float().cpu().numpy()
                lab = o["labels"].cpu().numpy() - 1
                keep = np.array([int(x) in cmap for x in lab], dtype=bool) if len(lab) else np.zeros(0, dtype=bool)
                return b[keep], s[keep], np.array([cmap[int(x)] for x in lab[keep]], dtype=int)

            report["splits"] = _det_splits(root, classes, predict)

    if not report["splits"]:
        raise RuntimeError("没有验证集或测试集。请先在「数据」页划分出验证集，再评估")
    (out / "eval.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    first = report["splits"].get("val") or next(iter(report["splits"].values()))
    emit("done", map50=first["map50"])
