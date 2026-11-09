"""误报测试：拿一段「不该报警」的视频（没有目标，或者只有容易认错的东西）跑模型，统计会误报多少次。

配置（JSON）：
    source       视频文件、RTSP / http 地址
    weights      best.pt 或导入的 model.onnx
    conf         置信度阈值
    fps          每秒检测几帧（默认 2）
    max_seconds  最多测多少秒（0 = 视频文件读完为止；直播流必须设）
    consecutive  连续几次检出才算一次报警（默认 1，和推理服务包的告警规则一致）
    cooldown     同一类别报警后冷却多少秒（默认 10）
    out          保存误报帧的目录（原图，方便加回数据集当负样本）
输出事件：
    {"type": "open", "fps", "duration"}
    {"type": "progress", "seconds", "total", "frames", "alarms"}
    {"type": "alarm", "t", "labels", "path", "preview"}      preview 是缩小后画了框的 JPEG（base64）
    {"type": "done", "seconds", "frames", "hitFrames", "alarms", "perHour", "perClass", "stopped"}
标准输入：{"cmd": "stop"} 提前结束（会正常输出统计）
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from . import models  # noqa: E402
from .protocol import emit, log  # noqa: E402

MAX_SAVED = 60


def _detector(weights: Path, device):
    """返回 (predict(PIL 图, conf) → [(x0, y0, x1, y1, 类别序号, 分数)], 类别名)"""
    if weights.suffix.lower() == ".onnx":
        from . import onnx_model

        det, info = onnx_model.load(weights)
        use = info.get("use")
        return (lambda img, conf: [d for d in det.predict(img, conf) if use is None or d[4] in use]), info.get("classes", [])
    model, ck = models.load(weights, device)
    if models.is_cls(ck["model"]):
        raise RuntimeError("分类模型没有框，不能做误报测试")
    from torchvision.transforms import functional as F

    @torch.no_grad()
    def predict(img, conf):
        o = model([F.to_tensor(img).to(device)])[0]
        res = []
        for b, s, l in zip(o["boxes"].tolist(), o["scores"].tolist(), o["labels"].tolist()):
            if s >= conf:
                res.append((*b, l - 1, s))
        return res

    return predict, ck["classes"]


def _preview(frame_bgr: np.ndarray, dets: list, classes: list[str]) -> str:
    h, w = frame_bgr.shape[:2]
    s = 360 / max(w, 1)
    img = cv2.resize(frame_bgr, (int(w * s), int(h * s)))
    for x0, y0, x1, y1, c, sc in dets:
        p0, p1 = (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s))
        cv2.rectangle(img, p0, p1, (0, 0, 255), 2)
        cv2.putText(img, f"{sc:.2f}", (p0[0], max(p0[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def run(cfg: dict) -> None:
    src = cfg["source"]
    weights = Path(cfg["weights"])
    conf = float(cfg.get("conf", 0.5))
    fps = max(0.2, float(cfg.get("fps", 2)))
    max_seconds = float(cfg.get("max_seconds", 0) or 0)
    consecutive = max(1, int(cfg.get("consecutive", 1)))
    cooldown = float(cfg.get("cooldown", 10))
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    stream = "://" in src
    if stream and max_seconds <= 0:
        raise RuntimeError("视频流没有结尾，请设置最长测试时间")

    stop = threading.Event()

    def read_stdin():
        for line in sys.stdin:
            try:
                if json.loads(line).get("cmd") == "stop":
                    stop.set()
                    return
            except Exception:
                pass

    threading.Thread(target=read_stdin, daemon=True).start()

    mps = getattr(torch.backends, "mps", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if (mps is not None and mps.is_available()) else "cpu")
    predict, classes = _detector(weights, device)
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("打不开视频，确认路径或地址正确")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (1 <= vfps <= 240):
        vfps = 25.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = 0.0 if stream else nframes / vfps
    total = max_seconds if (stream or (max_seconds and max_seconds < duration)) else duration
    emit("open", fps=vfps, duration=duration)
    log(f"开始误报测试：每秒检测 {fps:g} 帧，阈值 {conf:.2f}")

    step = max(1, round(vfps / fps))
    idx = 0
    frames = hit_frames = alarms = saved = 0
    streak: dict[int, int] = {}
    last_alarm: dict[int, float] = {}
    per_class: dict[int, int] = {}
    t0 = time.time()
    next_emit = 0.0
    seconds = 0.0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            if stream:
                time.sleep(0.2)
                if time.time() - t0 > max_seconds:
                    break
                continue
            break
        # 视频文件按帧号算时间；直播流按实际经过的时间
        seconds = (time.time() - t0) if stream else idx / vfps
        idx += 1
        if total and seconds >= total:
            break
        if not stream and (idx - 1) % step:
            continue
        if stream and frames and seconds < frames / fps:
            continue
        frames += 1
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets = predict(img, conf)
        if dets:
            hit_frames += 1
        hit = {int(d[4]) for d in dets}
        fired = []
        for c in set(streak) | hit:
            streak[c] = streak.get(c, 0) + 1 if c in hit else 0
            if streak[c] >= consecutive and seconds - last_alarm.get(c, -1e9) >= cooldown:
                last_alarm[c] = seconds
                per_class[c] = per_class.get(c, 0) + 1
                fired.append(c)
        if fired:
            alarms += len(fired)
            labels = [classes[c] if 0 <= c < len(classes) else str(c) for c in fired]
            path = ""
            if saved < MAX_SAVED:
                path = str(out / f"fp_{int(seconds * 1000):09d}.jpg")
                cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(path)
                saved += 1
            emit("alarm", t=round(seconds, 1), labels=labels, path=path, preview=_preview(frame, [d for d in dets if int(d[4]) in fired], classes))
        if seconds >= next_emit:
            next_emit = seconds + 1
            emit("progress", seconds=round(seconds, 1), total=round(total, 1), frames=frames, alarms=alarms)
    cap.release()
    hours = max(seconds, 1e-6) / 3600
    emit(
        "done",
        seconds=round(seconds, 1),
        frames=frames,
        hitFrames=hit_frames,
        alarms=alarms,
        perHour=round(alarms / hours, 1) if seconds > 0 else 0,
        perClass={(classes[c] if 0 <= c < len(classes) else str(c)): n for c, n in per_class.items()},
        stopped=stop.is_set(),
    )
