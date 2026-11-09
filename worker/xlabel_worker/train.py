"""训练：目标检测 / 实例分割 / 关键点 / 图像分类。

配置（JSON）：
    dataset   桌面端导出的数据集目录（分类是分类文件夹，其他任务是 COCO）
    out       训练输出目录（best.pt / last.pt / metrics.jsonl / eval.json）
    model     模型名，见 models.MODELS
    classes   类别名列表，顺序对应 category_id 1..n（分类对应 label 0..n-1）
    epochs, img_size, batch, lr, workers, device("auto"/"cpu"/"cuda")
    关键点另有：num_keypoints、keypoints（点名）、flip_index（左右翻转对应关系）

在 out 目录下创建 STOP 文件即可让训练在当前迭代结束后停止。

分类没有 mAP：metric 事件里 map50 = 准确率，map = 宏平均 F1（界面按任务改名显示）。
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import models
from .data import ClsFolder, CocoDetection, collate, has_cls_split, has_split
from .metrics import ImageGt, ImagePred, classify_eval, evaluate
from .protocol import emit, log


def pick_device(pref: str) -> torch.device:
    if pref != "cpu" and torch.cuda.is_available():
        return torch.device("cuda")
    # Apple 芯片的 Mac 用 MPS
    mps = getattr(torch.backends, "mps", None)
    if pref != "cpu" and mps is not None and mps.is_available():
        return torch.device("mps")
    if pref == "cuda":
        log("没有检测到可用的 NVIDIA 显卡，改用 CPU 训练")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return "Apple 芯片（MPS）" if device.type == "mps" else "CPU"


# ---------- 检测 / 分割 / 关键点 ----------


@torch.no_grad()
def predict_dataset(model, ds: CocoDetection, device, batch: int = 4):
    model.eval()
    preds, gts = [], []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=0)
    for imgs, tgts in loader:
        outs = model([i.to(device) for i in imgs])
        for o, t in zip(outs, tgts):
            preds.append(
                ImagePred(
                    boxes=o["boxes"].float().cpu().numpy(),
                    scores=o["scores"].float().cpu().numpy(),
                    labels=o["labels"].cpu().numpy() - 1,
                )
            )
            gts.append(ImageGt(boxes=t["boxes"].numpy(), labels=t["labels"].numpy() - 1, image_id=int(t["image_id"])))
    return preds, gts


class DetTask:
    """检测、分割、关键点共用：COCO 数据集 + torchvision 检测模型（训练时返回损失字典）"""

    def __init__(self, cfg: dict, root: Path, key: str):
        self.root = root
        self.key = key
        self.seg = models.is_seg(key)
        self.nk = int(cfg.get("num_keypoints", 17)) if models.is_pose(key) else 0
        self.flip = cfg.get("flip_index")
        self.keys = ("boxes", "labels") + (("masks",) if self.seg else ()) + (("keypoints",) if self.nk else ())

    def dataset(self, split: str, train: bool):
        return CocoDetection(self.root, split, train=train, seg=self.seg, num_keypoints=self.nk, flip_index=self.flip)

    def has_split(self, split: str) -> bool:
        return has_split(self.root, split)

    def collate(self, batch):
        return collate(batch)

    def loss(self, model, batch, device):
        imgs, tgts = batch
        imgs = [x.to(device, non_blocking=True) for x in imgs]
        tgts = [{k: v.to(device) for k, v in t.items() if k in self.keys} for t in tgts]
        return sum(model(imgs, tgts).values())

    def evaluate(self, model, ds, device, batch, n_classes):
        preds, gts = predict_dataset(model, ds, device, batch)
        return evaluate(preds, gts, n_classes)

    def save_extra(self) -> dict:
        return {"num_keypoints": self.nk} if self.nk else {}


# ---------- 图像分类 ----------


class ClsTask:
    def __init__(self, cfg: dict, root: Path, key: str):
        self.root = root
        self.size = int(cfg.get("img_size", 224))
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def dataset(self, split: str, train: bool):
        return ClsFolder(self.root, split, train=train, size=self.size)

    def has_split(self, split: str) -> bool:
        return has_cls_split(self.root, split)

    def collate(self, batch):
        imgs, labels, ids = zip(*batch)
        return torch.stack(imgs), torch.tensor(labels), list(ids)

    def loss(self, model, batch, device):
        imgs, labels, _ = batch
        return self.criterion(model(imgs.to(device, non_blocking=True)), labels.to(device))

    @torch.no_grad()
    def evaluate(self, model, ds, device, batch, n_classes):
        model.eval()
        loader = torch.utils.data.DataLoader(ds, batch_size=max(batch, 16), shuffle=False, collate_fn=self.collate, num_workers=0)
        y_true, y_pred, ids, scores = [], [], [], []
        for imgs, labels, idl in loader:
            prob = torch.softmax(model(imgs.to(device)).float(), dim=1).cpu()
            conf, pred = prob.max(dim=1)
            y_true += labels.tolist()
            y_pred += pred.tolist()
            scores += conf.tolist()
            ids += idl
        return classify_eval(y_true, y_pred, ids, n_classes, scores=np.array(scores))

    def save_extra(self) -> dict:
        return {}


# ---------- 训练主流程 ----------


def run(cfg: dict) -> None:
    root = Path(cfg["dataset"])
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    classes: list[str] = cfg["classes"]
    epochs = int(cfg.get("epochs", 30))
    key = cfg.get("model", "frcnn_mobilenet")
    cls = models.is_cls(key)
    # SSDLite 固定 320 输入
    img_size = 320 if models.kind_of(key) == "ssd" else int(cfg.get("img_size", 224 if cls else 640))
    cfg["img_size"] = img_size
    batch = int(cfg.get("batch", 4))
    lr = float(cfg.get("lr", 0.01))
    device = pick_device(cfg.get("device", "auto"))
    torch.manual_seed(int(cfg.get("seed", 0)))
    task = ClsTask(cfg, root, key) if cls else DetTask(cfg, root, key)

    log(f"设备：{device_name(device)}，模型：{models.MODELS[key]['name']}")
    train_ds = task.dataset("train", True)
    val_split = "val" if task.has_split("val") else "train"
    if val_split == "train":
        log("没有验证集，用训练集评估（指标会偏高）。建议划分出 10%~20% 作为验证集。")
    val_ds = task.dataset(val_split, False)
    log(f"训练集 {len(train_ds)} 张，验证集 {len(val_ds)} 张，类别 {len(classes)} 个")
    if len(train_ds) == 0:
        raise RuntimeError("训练集是空的")

    workers = int(cfg.get("workers", 0 if os.name == "nt" else 2))
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        collate_fn=task.collate,
        num_workers=workers,
        drop_last=len(train_ds) > batch,
    )
    model = models.build(key, len(classes), img_size, num_keypoints=int(cfg.get("num_keypoints", 17))).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    iters_per_epoch = max(len(loader), 1)
    total = epochs * iters_per_epoch
    warmup = min(300, iters_per_epoch * 2)

    def lr_factor(it: int) -> float:
        if it < warmup:
            return 0.001 + (1 - 0.001) * it / warmup
        p = (it - warmup) / max(total - warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    extra = task.save_extra()

    metrics_file = (out / "metrics.jsonl").open("a", encoding="utf-8")
    best = -1.0
    best_epoch = 0
    best_map50 = None
    stopped = False
    t0 = time.time()
    it = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for i, b in enumerate(loader):
            if stop_file.exists():
                stopped = True
                break
            with torch.autocast(device.type, enabled=use_amp):
                loss = task.loss(model, b, device)
            if not torch.isfinite(loss):
                log(f"第 {epoch} 轮出现异常损失，跳过这一批")
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            losses.append(loss.item())
            if i % max(1, iters_per_epoch // 20) == 0 or i == iters_per_epoch - 1:
                elapsed = time.time() - t0
                emit(
                    "progress",
                    epoch=epoch,
                    epochs=epochs,
                    iter=i + 1,
                    iters=iters_per_epoch,
                    loss=float(np.mean(losses[-20:])),
                    eta=elapsed / it * (total - it) if it else None,
                )
        if stopped:
            log("收到停止指令，已停止训练")
            break
        r = task.evaluate(model, val_ds, device, batch, len(classes))
        # 综合分：检测同 YOLO（mAP50 相同时选框更准的）；分类兼顾准确率和各类别均衡
        fitness = 0.5 * r.map50 + 0.5 * r.map if cls else 0.1 * r.map50 + 0.9 * r.map
        is_best = fitness > best
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else None,
            "map50": r.map50,
            "map": r.map,
            "precision": r.precision,
            "recall": r.recall,
            "lr": opt.param_groups[0]["lr"],
            "time": time.time() - t0,
            "best": is_best,
        }
        metrics_file.write(json.dumps(rec) + "\n")
        metrics_file.flush()
        emit("metric", **rec)
        models.save(out / "last.pt", model, key, classes, img_size, {"epoch": epoch, **extra})
        if is_best:
            best, best_epoch, best_map50 = fitness, epoch, r.map50
            models.save(out / "best.pt", model, key, classes, img_size, {"epoch": epoch, "map50": r.map50, **extra})
            if cls:
                log(f"第 {epoch} 轮准确率 {r.map50:.3f}、宏平均 F1 {r.map:.3f}，保存为最好的模型")
            else:
                log(f"第 {epoch} 轮 mAP50 {r.map50:.3f}、mAP50-95 {r.map:.3f}，保存为最好的模型")
    metrics_file.close()

    if not (out / "best.pt").exists():
        emit("done", stopped=stopped, bestMap50=None, bestEpoch=None)
        return

    # 用最好的模型做最终评估，给评估页用
    log("用最好的模型做最终评估…")
    model, _ = models.load(out / "best.pt", device)
    report = {"classes": classes, "bestEpoch": best_epoch, "task": models.task_of(key), "splits": {}}
    for split in ("val", "test"):
        if not task.has_split(split):
            continue
        ds = task.dataset(split, False)
        r = task.evaluate(model, ds, device, batch, len(classes))
        for c in r.per_class:
            c["name"] = classes[c["index"]]
        report["splits"][split] = {
            "images": len(ds),
            "map50": r.map50,
            "map": r.map,
            "precision": r.precision,
            "recall": r.recall,
            "perClass": r.per_class,
            "confusion": r.confusion,
            "errors": r.errors[:500],
        }
    (out / "eval.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    emit("done", stopped=stopped, bestMap50=best_map50, bestEpoch=best_epoch)
