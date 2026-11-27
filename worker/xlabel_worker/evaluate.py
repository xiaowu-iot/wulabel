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
from .metrics import (
    ImageGt,
    ImagePred,
    classify_eval,
    classify_threshold_curve,
    evaluate,
    multilabel_eval,
    multilabel_threshold_curve,
    threshold_curve,
)
from .i18n import tr
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
        n = len(ds.images)
        emit("progress", split=split, done=0, total=n)
        for i, im in enumerate(ds.images):
            boxes, scores, labels = predict(load_image(ds.dir / im["file_name"]))
            preds.append(ImagePred(boxes=boxes, scores=scores, labels=labels))
            t = ds.target(i)
            gts.append(ImageGt(boxes=t["boxes"].numpy(), labels=t["labels"].numpy() - 1, image_id=int(im["id"])))
            # 每 10 张和最后一张报一次进度
            if (i + 1) % 10 == 0 or i + 1 == n:
                emit("progress", split=split, done=i + 1, total=n)
        r = evaluate(preds, gts, len(classes))
        r.thresholds = threshold_curve(preds, gts, len(classes))
        out[split] = split_report(r, classes, len(ds.images))
        n, v = len(ds.images), f"{r.map50:.3f}"
        log(tr("验证集：{n} 张，mAP50 {map50}", n=n, map50=v) if split == "val" else tr("测试集：{n} 张，mAP50 {map50}", n=n, map50=v))
    return out


@torch.no_grad()
def _multi_splits(root: Path, classes: list[str], model, ck: dict, cmap: dict[int, int], task, device) -> dict:
    """多标签分类：每个类别的 sigmoid 分数；模型里没有的项目类别分数记 0"""
    out = {}
    model.eval()
    for split in SPLITS:
        if not has_cls_split(root, split):
            continue
        ds = ClsFolder(root, split, train=False, size=int(ck["img_size"]))
        ys, ps, ids = [], [], []
        emit("progress", split=split, done=0, total=len(ds))
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, collate_fn=task.collate)
        for imgs, labels, idl in loader:
            prob = torch.sigmoid(model(imgs.to(device)).float()).cpu().numpy()
            p = np.zeros((len(idl), len(classes)))
            for mi, pi in cmap.items():
                p[:, pi] = prob[:, mi]
            ps.append(p)
            ys.append(labels.numpy())
            ids += idl
            emit("progress", split=split, done=len(ids), total=len(ds))
        y = np.concatenate(ys) if ys else np.zeros((0, len(classes)))
        p = np.concatenate(ps) if ps else np.zeros((0, len(classes)))
        r = multilabel_eval(y, p, ids, len(classes))
        r.thresholds = multilabel_threshold_curve(y, p)
        out[split] = split_report(r, classes, len(ds))
    return out


def _onnx_cls_splits(root: Path, classes: list[str], det, cmap: dict[int, int], use, report: dict) -> dict:
    """ONNX 分类模型（单标签或多标签）在分类数据集上评估"""
    out = {}
    multi = bool(getattr(det.m, "multi", False))
    for split in SPLITS:
        if not has_cls_split(root, split):
            continue
        doc = json.loads((root / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
        if bool(doc.get("multiLabel")) != multi:
            raise RuntimeError(tr("这个模型和项目一个是多标签分类、一个是单标签分类，不能用来评估"))
        items = doc["images"]
        emit("progress", split=split, done=0, total=len(items))
        ys, ps, ids = [], [], []
        y_true, y_pred, scores = [], [], []
        for i, it in enumerate(items):
            res = det.predict_full(load_image(root / it["file"]), 0.0)
            if multi:
                p = np.zeros(len(classes))
                for d in res:
                    if d["cls"] in cmap and (use is None or d["cls"] in use):
                        p[cmap[d["cls"]]] = d["score"]
                y = np.zeros(len(classes))
                for c in it["labels"]:
                    y[int(c)] = 1.0
                ps.append(p)
                ys.append(y)
            else:
                top = next((d for d in res if use is None or d["cls"] in use), None)
                t = int(it["label"])
                pi = cmap.get(top["cls"], -1) if top else -1
                y_true.append(t)
                # 判成项目里没有的类别时记成分错
                y_pred.append(pi if pi >= 0 else (t + 1) % max(len(classes), 1))
                scores.append(top["score"] if top else 0.0)
            ids.append(int(it["id"]))
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                emit("progress", split=split, done=i + 1, total=len(items))
        if multi:
            y = np.stack(ys) if ys else np.zeros((0, len(classes)))
            p = np.stack(ps) if ps else np.zeros((0, len(classes)))
            r = multilabel_eval(y, p, ids, len(classes))
            r.thresholds = multilabel_threshold_curve(y, p)
            report["multiLabel"] = True
        else:
            r = classify_eval(y_true, y_pred, ids, len(classes), scores=np.array(scores))
            r.thresholds = classify_threshold_curve(y_true, y_pred, scores)
        out[split] = split_report(r, classes, len(items))
        n, v = len(items), f"{r.map50:.3f}"
        log(tr("验证集：{n} 张，准确率 {acc}", n=n, acc=v) if split == "val" else tr("测试集：{n} 张，准确率 {acc}", n=n, acc=v))
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
            raise RuntimeError(tr("模型的类别名和项目类别一个都对不上，请在导入模型时把类别名改成项目里的名字"))
        log(tr("模型的 {n} 个类别和项目类别对应上", n=len(cmap)))
        report["task"] = det.task
        if det.task == "classify":
            report["splits"] = _onnx_cls_splits(root, classes, det, cmap, use, report)
        else:

            def predict(img):
                dets = [d for d in det.predict(img, 0.05) if d[4] in cmap and (use is None or d[4] in use)]
                if not dets:
                    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
                a = np.array([d[:4] for d in dets], dtype=np.float32)
                return a, np.array([d[5] for d in dets]), np.array([cmap[d[4]] for d in dets], dtype=int)

            report["splits"] = _det_splits(root, classes, predict)
    else:
        model, ck = models.load(weights, device)
        key = ck["model"]
        report["task"] = models.task_of(key)
        cmap = _remap(ck["classes"], classes)
        if models.is_cls(key):
            task = ClsTask({"img_size": ck["img_size"]}, root, key)
            if bool(ck.get("multi_label")) != task.multi:
                raise RuntimeError(tr("这个模型和项目一个是多标签分类、一个是单标签分类，不能用来评估"))
            if task.multi:
                report["multiLabel"] = True
                report["splits"] = _multi_splits(root, classes, model, ck, cmap, task, device)
                for split in SPLITS:
                    if split in report["splits"]:
                        n, v = report["splits"][split]["images"], f"{report['splits'][split]['map50']:.3f}"
                        log(tr("验证集：{n} 张，mAP {map50}", n=n, map50=v) if split == "val" else tr("测试集：{n} 张，mAP {map50}", n=n, map50=v))
            for split in SPLITS if not task.multi else ():
                if not has_cls_split(root, split):
                    continue
                ds = ClsFolder(root, split, train=False, size=int(ck["img_size"]))
                y_true, y_pred, ids, scores = [], [], [], []
                model.eval()
                emit("progress", split=split, done=0, total=len(ds))
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
                        emit("progress", split=split, done=len(ids), total=len(ds))
                # 分错到项目里没有的类别时，混淆矩阵放不下，记成「判成第一个类以外」——这里归到最后统计前过滤
                yp = [p if p >= 0 else (t + 1) % max(len(classes), 1) for p, t in zip(y_pred, y_true)]
                r = classify_eval(y_true, yp, ids, len(classes), scores=np.array(scores))
                r.thresholds = classify_threshold_curve(y_true, yp, scores)
                report["splits"][split] = split_report(r, classes, len(ds))
                n, v = len(ds), f"{r.map50:.3f}"
                log(tr("验证集：{n} 张，准确率 {acc}", n=n, acc=v) if split == "val" else tr("测试集：{n} 张，准确率 {acc}", n=n, acc=v))
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
        raise RuntimeError(tr("没有验证集或测试集。请先在「数据」页划分出验证集，再评估"))
    (out / "eval.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    first = report["splits"].get("val") or next(iter(report["splits"].values()))
    emit("done", map50=first["map50"])
