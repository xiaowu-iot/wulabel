"""训练目标检测模型。

配置（JSON）：
    dataset   桌面端导出的 COCO 数据集目录
    out       训练输出目录（best.pt / last.pt / metrics.jsonl / eval.json）
    model     模型名，见 models.MODELS
    classes   类别名列表，顺序对应 category_id 1..n
    epochs, img_size, batch, lr, workers, device("auto"/"cpu"/"cuda")

在 out 目录下创建 STOP 文件即可让训练在当前迭代结束后停止。
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
from .data import CocoDetection, collate, has_split
from .metrics import ImageGt, ImagePred, evaluate
from .protocol import emit, log


def pick_device(pref: str) -> torch.device:
    if pref != "cpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cuda":
        log("没有检测到可用的 NVIDIA 显卡，改用 CPU 训练")
    return torch.device("cpu")


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


def run(cfg: dict) -> None:
    root = Path(cfg["dataset"])
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    classes: list[str] = cfg["classes"]
    epochs = int(cfg.get("epochs", 30))
    img_size = int(cfg.get("img_size", 640))
    batch = int(cfg.get("batch", 4))
    lr = float(cfg.get("lr", 0.01))
    key = cfg.get("model", "frcnn_mobilenet")
    seg = models.is_seg(key)
    device = pick_device(cfg.get("device", "auto"))
    torch.manual_seed(int(cfg.get("seed", 0)))

    log(f"设备：{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}，模型：{models.MODELS[key]['name']}")
    train_ds = CocoDetection(root, "train", train=True, seg=seg)
    val_split = "val" if has_split(root, "val") else "train"
    if val_split == "train":
        log("没有验证集，用训练集评估（指标会偏高）。建议划分出 10%~20% 作为验证集。")
    val_ds = CocoDetection(root, val_split, train=False, seg=seg)
    log(f"训练集 {len(train_ds)} 张，验证集 {len(val_ds)} 张，类别 {len(classes)} 个")
    if len(train_ds) == 0:
        raise RuntimeError("训练集是空的")

    workers = int(cfg.get("workers", 0 if os.name == "nt" else 2))
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=workers, drop_last=len(train_ds) > batch
    )
    model = models.build(key, len(classes), img_size).to(device)
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
        for i, (imgs, tgts) in enumerate(loader):
            if stop_file.exists():
                stopped = True
                break
            imgs = [x.to(device, non_blocking=True) for x in imgs]
            tgts = [{k: v.to(device) for k, v in t.items() if k in ("boxes", "labels", "masks")} for t in tgts]
            with torch.autocast(device.type, enabled=use_amp):
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())
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
        preds, gts = predict_dataset(model, val_ds, device, batch)
        r = evaluate(preds, gts, len(classes))
        # 综合分（同 YOLO）：mAP50 相同时选框更准的
        fitness = 0.1 * r.map50 + 0.9 * r.map
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
        models.save(out / "last.pt", model, key, classes, img_size, {"epoch": epoch})
        if is_best:
            best, best_epoch, best_map50 = fitness, epoch, r.map50
            models.save(out / "best.pt", model, key, classes, img_size, {"epoch": epoch, "map50": r.map50})
            log(f"第 {epoch} 轮 mAP50 {r.map50:.3f}、mAP50-95 {r.map:.3f}，保存为最好的模型")
    metrics_file.close()

    if not (out / "best.pt").exists():
        emit("done", stopped=stopped, bestMap50=None, bestEpoch=None)
        return

    # 用最好的模型做最终评估，给评估页用
    log("用最好的模型做最终评估…")
    model, _ = models.load(out / "best.pt", device)
    report = {"classes": classes, "bestEpoch": best_epoch, "splits": {}}
    for split in ("val", "test"):
        if not has_split(root, split):
            continue
        preds, gts = predict_dataset(model, CocoDetection(root, split, train=False, seg=seg), device, batch)
        r = evaluate(preds, gts, len(classes))
        for c in r.per_class:
            c["name"] = classes[c["index"]]
        report["splits"][split] = {
            "images": len(gts),
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
