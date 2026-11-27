"""Ultralytics 训练引擎：YOLOv8 / YOLO11 / YOLO12 / YOLO26 / YOLOv10 / YOLOv9 / YOLOv5u / RT-DETR 等，
以及用户自己的 .pt（接着训）、.yaml（从头训）。

和内置模型用同一套事件（progress / metric / batch / done），训完自动：导出 ONNX → model.onnx + model.json
→ 用 Wulabel 自己的评估跑验证集 / 测试集写 eval.json。之后预标注、评估、部署都按 ONNX 模型处理。

配置（桌面端写）：
    dataset     评估用的数据集（COCO；分类是分类文件夹 + 清单）
    data_yolo   训练用的 YOLO 格式数据集（检测、分割、关键点）；分类直接用 dataset
    out         训练记录目录；Ultralytics 的产物在 out/native/
    model       模型名（yolo11s-seg.pt 等，自动下载）或本地 .pt / .yaml 的完整路径
    init_weights 接着之前某次 Ultralytics 训练：那次的 native/weights/best.pt
    resume      从断点继续（native/weights/last.pt）
    classes、task、epochs、img_size、batch（-1 表示自动）、device、workers、extra（原样传给训练的参数）
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.request
from pathlib import Path

from .i18n import tr
from .protocol import emit, log

# 下载官方权重：按发布版本依次找（新模型在新版本里）
ASSET_TAGS = ["v8.4.0", "v8.3.0", "v8.2.0", "v8.1.0", "v8.0.0"]
ASSET_URL = "https://github.com/ultralytics/assets/releases/download/{tag}/{name}"
# 不允许界面「更多参数」改掉的：路径、输出位置由 Wulabel 管
RESERVED = {"data", "project", "name", "exist_ok", "resume", "model", "task", "mode"}


class _Stop(Exception):
    """界面点了停止"""


def _weights_dir() -> Path:
    th = os.environ.get("TORCH_HOME")
    root = Path(th).parent if th else Path.home() / ".cache" / "wulabel"
    d = root / "ultralytics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Wulabel"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status < 400
    except Exception:
        return False


def resolve_model(model: str) -> str:
    """本地文件原样用；官方模型名（yolo11s-seg.pt）下载到训练环境目录，下过的直接用；结构文件名（yolo11s.yaml）交给 Ultralytics"""
    p = Path(model)
    if p.is_file():
        return str(p)
    if model.lower().endswith((".yaml", ".yml")):
        return model
    name = model if model.lower().endswith(".pt") else model + ".pt"
    dest = _weights_dir() / name
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    from .download import fetch

    for tag in ASSET_TAGS:
        url = ASSET_URL.format(tag=tag, name=name)
        if _url_exists(url):
            fetch(url, dest, name)
            return str(dest)
    raise RuntimeError(
        tr("找不到官方模型 {name}，请检查名字，或者把权重文件下载到本地后用「本地模型文件」选择", name=name)
    )


def _forward_logs() -> None:
    """Ultralytics 的日志转成 Wulabel 的日志事件（不直接打到标准输出，免得混进事件流）"""

    class H(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record).strip()
            except Exception:
                return
            # 去掉终端颜色
            import re

            msg = re.sub(r"\x1b\[[0-9;]*m", "", msg)
            if msg:
                log(msg)

    lg = logging.getLogger("ultralytics")
    for h in list(lg.handlers):
        lg.removeHandler(h)
    h = H()
    h.setLevel(logging.INFO)
    lg.addHandler(h)
    lg.propagate = False


def _device(want: str):
    import torch

    if want == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _prepare_data(cfg: dict, task: str, out: Path) -> str:
    """训练用的数据：检测/分割/关键点写一份带绝对路径的 data.yaml（没有验证集时用训练集验证）；分类用分类文件夹"""
    if task == "classify":
        root = Path(cfg["dataset"])
        if not (root / "val").exists():
            # Ultralytics 分类必须有 val：没有验证集时用训练集
            shutil.copytree(root / "train", root / "val", copy_function=_link_or_copy)
        return str(root)
    root = Path(cfg["data_yolo"]).resolve()
    splits = {s: (root / "images" / s).exists() for s in ("train", "val", "test")}
    lines = [f"path: {json.dumps(str(root))}", "train: images/train", f"val: images/{'val' if splits['val'] else 'train'}"]
    if splits["test"]:
        lines.append("test: images/test")
    src = (root / "data.yaml").read_text(encoding="utf-8").splitlines()
    keep = False
    for ln in src:
        # 关键点形状、翻转顺序、类别名原样保留
        if ln.startswith(("kpt_shape:", "flip_idx:", "nc:")):
            lines.append(ln)
        if ln.startswith("names:"):
            keep = True
        if keep:
            lines.append(ln)
    yaml = out / "data_ultralytics.yaml"
    yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(yaml)


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _parse_extra(extra) -> dict:
    """「更多参数」：{key: value} 或 "key=value" 一行一个；值按 Python / YAML 习惯转成数字、布尔"""
    if isinstance(extra, dict):
        items = extra.items()
    else:
        items = []
        for ln in str(extra or "").splitlines():
            ln = ln.split("#", 1)[0].strip()
            if "=" in ln:
                k, v = ln.split("=", 1)
                items.append((k.strip(), v.strip()))
    out = {}
    for k, v in items:
        if not k or k in RESERVED:
            continue
        if isinstance(v, str):
            low = v.lower()
            if low in ("true", "false"):
                v = low == "true"
            elif low in ("none", "null"):
                v = None
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        v = v.strip("'\"")
        out[k] = v
    return out


def _metrics(trainer, task: str) -> dict:
    m = getattr(trainer, "metrics", None) or {}
    if task == "classify":
        top1 = float(m.get("metrics/accuracy_top1", 0) or 0)
        return {"map50": top1, "map": top1, "precision": top1, "recall": top1}
    # 分割、关键点也按框的指标（和内置模型、Wulabel 评估一致）
    return {
        "map50": float(m.get("metrics/mAP50(B)", 0) or 0),
        "map": float(m.get("metrics/mAP50-95(B)", 0) or 0),
        "precision": float(m.get("metrics/precision(B)", 0) or 0),
        "recall": float(m.get("metrics/recall(B)", 0) or 0),
    }


def _loss(trainer) -> float:
    t = getattr(trainer, "tloss", None)
    try:
        return float(t.sum()) if hasattr(t, "sum") else float(t or 0)
    except Exception:
        return 0.0


def _callbacks(model, out: Path, task: str, epochs: int, state: dict) -> None:
    stop_file = out / "STOP"

    def on_epoch_start(trainer):
        state["iter"] = 0
        state["t_epoch"] = time.time()

    def on_batch_end(trainer):
        if stop_file.exists():
            raise _Stop()
        state["iter"] = state.get("iter", 0) + 1
        iters = len(getattr(trainer, "train_loader", []) or []) or 1
        now = time.time()
        if now - state.get("last", 0) < 0.5 and state["iter"] < iters:
            return
        state["last"] = now
        ep = int(trainer.epoch) + 1
        done = (ep - 1) * iters + state["iter"]
        elapsed = now - state["t0"]
        eta = elapsed / max(done - state.get("done0", 0), 1) * (epochs * iters - done)
        emit("progress", epoch=ep, epochs=epochs, iter=state["iter"], iters=iters, loss=round(_loss(trainer), 4), eta=int(eta))

    def on_fit_epoch_end(trainer):
        ep = int(trainer.epoch) + 1
        # 训练结束时 Ultralytics 用 best.pt 再验证一遍，也会触发这个回调：同一轮只记一次
        if state.get("metric_epoch") == ep:
            return
        state["metric_epoch"] = ep
        m = _metrics(trainer, task)
        fit, best_fit = getattr(trainer, "fitness", None), getattr(trainer, "best_fitness", None)
        best = fit is not None and best_fit is not None and float(fit) >= float(best_fit)
        lr = next(iter((getattr(trainer, "lr", None) or {}).values()), 0)
        if best:
            state["best_epoch"] = ep
            state["best_map50"] = m["map50"]
        line = {"epoch": ep, "loss": round(_loss(trainer), 5), **{k: round(v, 5) for k, v in m.items()},
                "lr": float(lr or 0), "time": round(time.time() - state.get("t_epoch", time.time()), 1), "best": bool(best)}
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        emit("metric", **line)

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def _cls_names(cfg: dict, names: list[str]) -> list[str]:
    """分类：Ultralytics 的类别是文件夹名（按字母排序），换回项目类别名"""
    root = Path(cfg["dataset"])
    classes = cfg["classes"]
    folder = {}
    for s in ("train", "val", "test"):
        f = root / "annotations" / f"{s}.json"
        if f.exists():
            for it in json.loads(f.read_text(encoding="utf-8"))["images"]:
                parts = it["file"].split("/")
                if len(parts) >= 3 and "label" in it:
                    folder[parts[1]] = classes[int(it["label"])]
    return [folder.get(n, n) for n in names]


def _export(best: Path, out: Path, cfg: dict, task: str, size: int) -> dict:
    from ultralytics import YOLO

    from . import onnx_decode

    log(tr("导出 ONNX…"))
    m = YOLO(str(best))
    path = Path(m.export(format="onnx", imgsz=size, simplify=False, dynamic=False, verbose=False))
    dst = out / "model.onnx"
    shutil.move(str(path), dst)
    names = [str(m.names[i]) for i in sorted(m.names)] if isinstance(m.names, dict) else list(m.names)
    if task == "classify":
        names = _cls_names(cfg, names)
    info = onnx_decode.inspect_session(onnx_decode.session(dst, use_gpu=False))
    meta = {
        "kind": info["kind"], "task": task, "classes": names, "imgSize": size, "kptShape": info.get("kptShape"),
        "engine": "ultralytics", "model": cfg.get("model"),
    }
    (out / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run(cfg: dict) -> None:
    # Ultralytics 不要自己 pip 装东西、不要上报使用统计；设置文件放训练环境目录
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(_weights_dir() / "config"))
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(tr("没有安装 Ultralytics 组件，请重新开始训练（会自动安装）")) from e
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"sync": False, "weights_dir": str(_weights_dir())})
    except Exception:
        pass
    _forward_logs()
    import torch

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    task = cfg["task"]
    if task == "multilabel":
        raise RuntimeError(tr("Ultralytics 的分类模型不支持多标签分类，请选内置模型"))
    epochs = int(cfg.get("epochs", 100))
    size = max(32, round(int(cfg.get("img_size", 640)) / 32) * 32)
    native = out / "native"
    last = native / "weights" / "last.pt"
    resume = bool(cfg.get("resume")) and last.exists()
    data = _prepare_data(cfg, task, out)
    extra = _parse_extra(cfg.get("extra"))
    state = {"t0": time.time()}

    while True:
        batch = int(cfg.get("batch", 16))
        if resume:
            weights = str(last)
        else:
            weights = cfg.get("init_weights") or resolve_model(str(cfg["model"]))
        log(tr("模型：{w}", w=weights))
        model = YOLO(weights, task=task)
        if getattr(model, "task", task) != task:
            raise RuntimeError(tr("这个模型是「{m}」任务，和项目的任务类型不一致", m=model.task))
        _callbacks(model, out, task, epochs, state)
        args = {
            "data": data, "epochs": epochs, "imgsz": size, "batch": batch if batch > 0 else -1, "device": _device(cfg.get("device", "auto")),
            "workers": int(cfg.get("workers", 0 if os.name == "nt" else 2)), "project": str(out), "name": "native",
            "exist_ok": True, "plots": False, "verbose": False, "seed": int(cfg.get("seed", 0)),
        }
        args.update(extra)
        stopped = False
        try:
            if resume:
                model.train(resume=True)
            else:
                model.train(**args)
        except _Stop:
            stopped = True
            log(tr("已停止训练"))
        except Exception as e:  # noqa: BLE001
            from .train import free_memory, is_oom

            if is_oom(e) and batch > 1 and not (out / "STOP").exists():
                del model
                free_memory()
                new = max(1, batch // 2)
                log(tr("显存不够，批大小从 {batch} 自动调到 {new}，从头重新训练", batch=batch, new=new))
                emit("batch", batch=new, was=batch)
                cfg["batch"] = new
                resume = False
                shutil.rmtree(native, ignore_errors=True)
                (out / "metrics.jsonl").unlink(missing_ok=True)
                continue
            raise
        break

    best = native / "weights" / "best.pt"
    if not best.exists():
        best = last
    if not best.exists():
        if stopped:
            emit("done", stopped=True, bestMap50=None, bestEpoch=None)
            return
        raise RuntimeError(tr("训练没有产出模型"))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    meta = _export(best, out, cfg, task, size)

    # 用 Wulabel 自己的评估：和内置模型同一套指标、错例和推荐阈值
    from . import evaluate

    try:
        evaluate.run({"dataset": cfg["dataset"], "out": str(out), "weights": str(out / "model.onnx"), "classes": cfg["classes"],
                      "quiet": True})
        ev_path = out / "eval.json"
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        ev.pop("reevaluated", None)
        ev["bestEpoch"] = state.get("best_epoch")
        ev_path.write_text(json.dumps(ev, ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log(tr("评估没有完成：{e}", e=e))
    emit("done", stopped=stopped, bestMap50=state.get("best_map50"), bestEpoch=state.get("best_epoch"), classes=meta["classes"])
