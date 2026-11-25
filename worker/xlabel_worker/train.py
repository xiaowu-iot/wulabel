"""训练：目标检测 / 实例分割 / 关键点 / 图像分类。

配置（JSON）：
    dataset   桌面端导出的数据集目录（分类是分类文件夹，其他任务是 COCO）
    out       训练输出目录（best.pt / last.pt / metrics.jsonl / eval.json）
    model     模型名，见 models.MODELS
    classes   类别名列表，顺序对应 category_id 1..n（分类对应 label 0..n-1）
    epochs, img_size, batch, lr, workers, device("auto"/"cpu"/"cuda")
    关键点另有：num_keypoints、keypoints（点名）、flip_index（左右翻转对应关系）

在 out 目录下创建 STOP 文件即可让训练在当前迭代结束后停止。

显存（或内存）不够时自动把批大小减半、从头再训，直到批大小为 1；
每次调整会发一个 batch 事件（batch=新的批大小，was=原来的）。

分类没有 mAP：metric 事件里 map50 = 准确率，map = 宏平均 F1（界面按任务改名显示）。
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import models
from .data import ClsFolder, CocoDetection, cls_multi_label, collate, has_cls_split, has_split
from .i18n import tr
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
from .protocol import emit, log, warn


def pick_device(pref: str) -> torch.device:
    if pref != "cpu" and torch.cuda.is_available():
        return torch.device("cuda")
    # Apple 芯片的 Mac 用 MPS
    mps = getattr(torch.backends, "mps", None)
    if pref != "cpu" and mps is not None and mps.is_available():
        return torch.device("mps")
    if pref == "cuda":
        log(tr("没有检测到可用的 NVIDIA 显卡，改用 CPU 训练"))
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return tr("Apple 芯片（MPS）") if device.type == "mps" else "CPU"


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
        self.aug = cfg.get("aug")
        self.keys = ("boxes", "labels") + (("masks",) if self.seg else ()) + (("keypoints",) if self.nk else ())

    def dataset(self, split: str, train: bool):
        return CocoDetection(
            self.root, split, train=train, seg=self.seg, num_keypoints=self.nk, flip_index=self.flip, aug=self.aug
        )

    def has_split(self, split: str) -> bool:
        return has_split(self.root, split)

    def collate(self, batch):
        return collate(batch)

    def loss(self, model, batch, device):
        imgs, tgts = batch
        imgs = [x.to(device, non_blocking=True) for x in imgs]
        tgts = [{k: v.to(device) for k, v in t.items() if k in self.keys} for t in tgts]
        return sum(model(imgs, tgts).values())

    def evaluate(self, model, ds, device, batch, n_classes, thresholds: bool = False):
        preds, gts = predict_dataset(model, ds, device, batch)
        r = evaluate(preds, gts, n_classes)
        if thresholds:
            r.thresholds = threshold_curve(preds, gts, n_classes)
        return r

    def save_extra(self) -> dict:
        return {"num_keypoints": self.nk} if self.nk else {}


# ---------- 图像分类 ----------


class ClsTask:
    def __init__(self, cfg: dict, root: Path, key: str):
        self.root = root
        self.size = int(cfg.get("img_size", 224))
        self.aug = cfg.get("aug")
        # 多标签：每个类别各自判断有没有（sigmoid），一张图可以有好几个类别
        self.multi = cls_multi_label(root)
        self.criterion = torch.nn.BCEWithLogitsLoss() if self.multi else torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def dataset(self, split: str, train: bool):
        return ClsFolder(self.root, split, train=train, size=self.size, aug=self.aug)

    def has_split(self, split: str) -> bool:
        return has_cls_split(self.root, split)

    def collate(self, batch):
        imgs, labels, ids = zip(*batch)
        return torch.stack(imgs), torch.stack(labels) if self.multi else torch.tensor(labels), list(ids)

    def loss(self, model, batch, device):
        imgs, labels, _ = batch
        return self.criterion(model(imgs.to(device, non_blocking=True)), labels.to(device))

    @torch.no_grad()
    def evaluate(self, model, ds, device, batch, n_classes, thresholds: bool = False):
        model.eval()
        loader = torch.utils.data.DataLoader(ds, batch_size=max(batch, 16), shuffle=False, collate_fn=self.collate, num_workers=0)
        if self.multi:
            ys, ps, ids = [], [], []
            for imgs, labels, idl in loader:
                ps.append(torch.sigmoid(model(imgs.to(device)).float()).cpu().numpy())
                ys.append(labels.numpy())
                ids += idl
            y = np.concatenate(ys) if ys else np.zeros((0, n_classes))
            p = np.concatenate(ps) if ps else np.zeros((0, n_classes))
            r = multilabel_eval(y, p, ids, n_classes)
            if thresholds:
                r.thresholds = multilabel_threshold_curve(y, p)
            return r
        y_true, y_pred, ids, scores = [], [], [], []
        for imgs, labels, idl in loader:
            prob = torch.softmax(model(imgs.to(device)).float(), dim=1).cpu()
            conf, pred = prob.max(dim=1)
            y_true += labels.tolist()
            y_pred += pred.tolist()
            scores += conf.tolist()
            ids += idl
        r = classify_eval(y_true, y_pred, ids, n_classes, scores=np.array(scores))
        if thresholds:
            r.thresholds = classify_threshold_curve(y_true, y_pred, scores)
        return r

    def save_extra(self) -> dict:
        return {"multi_label": True} if self.multi else {}


def split_report(r, classes: list[str], images: int) -> dict:
    """评估结果写进 eval.json 的格式（训练结束时、重新评估时共用）"""
    for c in r.per_class:
        c["name"] = classes[c["index"]]
    for c in r.thresholds.get("perClass", []):
        c["name"] = classes[c["index"]]
    return {
        "images": images,
        "map50": r.map50,
        "map": r.map,
        "precision": r.precision,
        "recall": r.recall,
        "perClass": r.per_class,
        "confusion": r.confusion,
        "errors": r.errors[:500],
        "thresholds": r.thresholds,
        **({"exact": r.exact} if r.exact is not None else {}),
    }


# ---------- 训练主流程 ----------


class ZeroWatch:
    """主指标（mAP50 / 分类的准确率）连续 n 轮都是 0 时提示一次"""

    def __init__(self, n: int):
        self.n = max(1, n)
        self.streak = 0
        self.fired = False

    def update(self, value: float) -> bool:
        """记一轮的指标，该提示时返回 True（只返回一次）"""
        self.streak = self.streak + 1 if value <= 1e-6 else 0
        if self.streak >= self.n and not self.fired:
            self.fired = True
            return True
        return False


def save_resume(path: Path, state: dict) -> None:
    """断点先写临时文件再改名，存到一半被强制结束也不会留下坏文件"""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def is_oom(e: BaseException) -> bool:
    """显卡（CUDA / Apple MPS）或内存不够"""
    if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def run(cfg: dict) -> None:
    out = Path(cfg["out"])
    while True:
        batch = int(cfg.get("batch", 4))
        try:
            return run_once(cfg)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e) or batch <= 1 or (out / "STOP").exists():
                raise
        # 出了 except 再清显存：异常的调用栈还引用着上一次的张量
        free_memory()
        new = max(1, batch // 2)
        log(tr("显存不够，批大小从 {batch} 自动调到 {new}，从头重新训练", batch=batch, new=new))
        emit("batch", batch=new, was=batch)
        cfg["resume"] = False  # 批大小变了，断点作废，从头训
        for f in ("metrics.jsonl", "best.pt", "last.pt", "eval.json", "resume.pt"):
            try:
                (out / f).unlink(missing_ok=True)
            except OSError:
                pass
        cfg["batch"] = new


def run_once(cfg: dict) -> None:
    root = Path(cfg["dataset"])
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    # 从断点继续：上次每轮结束时存的模型、优化器、学习率进度等
    resume_path = out / "resume.pt"
    state = torch.load(resume_path, map_location="cpu", weights_only=False) if cfg.get("resume") and resume_path.exists() else None
    if state is not None:
        cfg["batch"] = int(state.get("batch", cfg.get("batch", 4)))  # 显存不够自动调小过的话，沿用调小后的
    classes: list[str] = cfg["classes"]
    epochs = int(cfg.get("epochs", 30))
    key = cfg.get("model", "frcnn_mobilenet")
    cls = models.is_cls(key)
    yolox = models.kind_of(key) == "yolox"
    # SSDLite 固定 320 输入；YOLOX 要 32 的倍数
    img_size = 320 if models.kind_of(key) == "ssd" else int(cfg.get("img_size", 224 if cls else 640))
    if yolox:
        img_size = max(32, round(img_size / 32) * 32)
    cfg["img_size"] = img_size
    batch = int(cfg.get("batch", 4))
    lr = float(cfg.get("lr", 0.01))
    device = pick_device(cfg.get("device", "auto"))
    torch.manual_seed(int(cfg.get("seed", 0)))
    if yolox:
        from .yolox_det import YoloxTask

        task = YoloxTask(cfg, root, key, epochs)
    else:
        task = ClsTask(cfg, root, key) if cls else DetTask(cfg, root, key)

    log(tr("设备：{device}，模型：{model}", device=device_name(device), model=models.MODELS[key]["name"]))
    train_ds = task.dataset("train", True)
    val_split = "val" if task.has_split("val") else "train"
    if val_split == "train":
        log(tr("没有验证集，用训练集评估（指标会偏高）。建议划分出 10%~20% 作为验证集。"))
    val_ds = task.dataset(val_split, False)
    log(tr("训练集 {train} 张，验证集 {val} 张，类别 {classes} 个", train=len(train_ds), val=len(val_ds), classes=len(classes)))
    if len(train_ds) == 0:
        raise RuntimeError(tr("训练集是空的"))

    workers = int(cfg.get("workers", 0 if os.name == "nt" else 2))
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        collate_fn=task.collate,
        num_workers=workers,
        drop_last=len(train_ds) > batch,
    )
    # 接着上次的模型训练时，预训练权重会被覆盖，没下载到也不用提示
    model = models.build(
        key,
        len(classes),
        img_size,
        pretrained=state is None,  # 接着断点训练时权重从断点读，不用再下预训练权重
        num_keypoints=int(cfg.get("num_keypoints", 17)),
        warn_missing=not cfg.get("init_weights"),
    ).to(device)
    if cfg.get("init_weights") and state is None:
        models.load_init(model, key, cfg["init_weights"], classes)
    # 多尺度训练：输入尺寸在几档之间随机（评估时 torchvision 用最后一档，也就是原尺寸）
    scale = float((cfg.get("aug") or {}).get("scale") or 0)
    if scale and models.kind_of(key) in ("rcnn", "retinanet", "fcos"):
        from . import augment

        sizes = augment.multiscale_sizes(img_size, scale)
        model.transform.min_size = sizes
        log(tr("多尺度训练：{sizes}", sizes=tr("、").join(str(s) for s in sizes)))
    # 任务可以自带优化器、学习率曲线和 EMA（YOLOX 按官方做法），没有的用通用的
    if hasattr(task, "optimizer"):
        opt, params = task.optimizer(model, lr, batch)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    iters_per_epoch = max(len(loader), 1)
    total = epochs * iters_per_epoch
    log(tr("批大小 {batch}，每轮 {n} 步，共 {m} 步", batch=batch, n=iters_per_epoch, m=total))
    warmup = min(300, iters_per_epoch * 2)

    def lr_factor(it: int) -> float:
        if it < warmup:
            return 0.001 + (1 - 0.001) * it / warmup
        p = (it - warmup) / max(total - warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    if hasattr(task, "lr_factor"):
        lr_factor = task.lr_factor(iters_per_epoch, total)  # noqa: F811
    if hasattr(task, "on_model"):
        task.on_model(model)
    clip = getattr(task, "clip", 10.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    extra = task.save_extra()

    metrics_file = (out / "metrics.jsonl").open("a", encoding="utf-8")
    best = -1.0
    best_epoch = 0
    best_map50 = None
    stopped = False
    zero = ZeroWatch(min(10, epochs))
    t0 = time.time()
    it = 0
    start_epoch = 1
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        if use_amp and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if hasattr(task, "load_resume"):
            task.load_resume(state.get("task") or {}, model, state["epoch"] + 1)
        it = int(state["it"])
        best, best_epoch, best_map50 = state["best"], state["best_epoch"], state["best_map50"]
        zero.streak, zero.fired = state.get("zero", (0, False))
        t0 = time.time() - float(state.get("elapsed", 0))
        start_epoch = int(state["epoch"]) + 1
        # 上次存完这一轮的记录、还没来得及存断点就停了：多出来的轮次删掉，免得曲线重复
        metrics_file.close()
        kept = [ln for ln in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip() and json.loads(ln)["epoch"] < start_epoch]
        (out / "metrics.jsonl").write_text("".join(ln + "\n" for ln in kept), encoding="utf-8")
        metrics_file = (out / "metrics.jsonl").open("a", encoding="utf-8")
        log(tr("从第 {epoch} 轮接着训练（共 {epochs} 轮）", epoch=start_epoch, epochs=epochs))
    for epoch in range(start_epoch, epochs + 1):
        if hasattr(task, "on_epoch"):
            task.on_epoch(epoch, model)
        model.train()
        losses = []
        for i, b in enumerate(loader):
            if stop_file.exists():
                stopped = True
                break
            with torch.autocast(device.type, enabled=use_amp):
                loss = task.loss(model, b, device)
            if not torch.isfinite(loss):
                log(tr("第 {epoch} 轮出现异常损失，跳过这一批", epoch=epoch))
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if hasattr(task, "after_step"):
                task.after_step(model)
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
            log(tr("收到停止指令，已停止训练"))
            break
        # 有 EMA 的任务用平滑后的权重评估和保存
        eval_model = task.eval_model(model) if hasattr(task, "eval_model") else model
        r = task.evaluate(eval_model, val_ds, device, batch, len(classes))
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
        models.save(out / "last.pt", eval_model, key, classes, img_size, {"epoch": epoch, **extra})
        if is_best:
            best, best_epoch, best_map50 = fitness, epoch, r.map50
            models.save(out / "best.pt", eval_model, key, classes, img_size, {"epoch": epoch, "map50": r.map50, **extra})
            if fitness <= 0:
                # 指标全是 0（多半是刚开始训）：照样存一份，免得没有模型可用，但不说成「最好的模型」
                log(tr("第 {epoch} 轮指标还是 0，先保存当前模型，之后有更好的会替换", epoch=epoch))
            elif cls and getattr(task, "multi", False):
                log(tr("第 {epoch} 轮 mAP {map50}、宏平均 F1 {f1}，保存为最好的模型", epoch=epoch, map50=f"{r.map50:.3f}", f1=f"{r.map:.3f}"))
            elif cls:
                log(tr("第 {epoch} 轮准确率 {acc}、宏平均 F1 {f1}，保存为最好的模型", epoch=epoch, acc=f"{r.map50:.3f}", f1=f"{r.map:.3f}"))
            else:
                log(tr("第 {epoch} 轮 mAP50 {map50}、mAP50-95 {map}，保存为最好的模型", epoch=epoch, map50=f"{r.map50:.3f}", map=f"{r.map:.3f}"))
        if zero.update(r.map50):
            text = tr(
                "连续 {n} 轮{metric}都是 0，模型基本没学到东西。常见原因：图太少或训练步数太少；标注或类别有问题；没加载到预训练权重（看日志开头）",
                n=zero.n,
                metric=tr("准确率") if cls else " mAP50 ",
            )
            # 中文里 mAP50 两边要空格，别的语言会多出空格，合并掉
            warn("zero_metric", " ".join(text.split()))
        # 断点：停止、关软件、崩溃后可以从这一轮之后接着训练
        save_resume(
            resume_path,
            {
                "epoch": epoch,
                "batch": batch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "task": task.resume_state() if hasattr(task, "resume_state") else None,
                "it": it,
                "best": best,
                "best_epoch": best_epoch,
                "best_map50": best_map50,
                "zero": (zero.streak, zero.fired),
                "elapsed": time.time() - t0,
            },
        )
    metrics_file.close()

    if not (out / "best.pt").exists():
        emit("done", stopped=stopped, bestMap50=None, bestEpoch=None)
        return

    # 用最好的模型做最终评估，给评估页用
    log(tr("用最好的模型做最终评估…"))
    model, _ = models.load(out / "best.pt", device)
    report = {"classes": classes, "bestEpoch": best_epoch, "task": models.task_of(key), "splits": {}}
    if getattr(task, "multi", False):
        report["multiLabel"] = True
    for split in ("val", "test"):
        if not task.has_split(split):
            continue
        ds = task.dataset(split, False)
        r = task.evaluate(model, ds, device, batch, len(classes), thresholds=True)
        report["splits"][split] = split_report(r, classes, len(ds))
    (out / "eval.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    if not stopped:
        # 训完了，断点用不上了（大模型的断点有几百 MB）
        resume_path.unlink(missing_ok=True)
    emit("done", stopped=stopped, bestMap50=best_map50, bestEpoch=best_epoch)
