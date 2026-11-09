"""实时验证：读取 RTSP 流或视频文件，用模型逐帧检测。

配置（JSON）：source（RTSP 地址或视频文件路径）、weights、conf、max_fps、preview_width
输出事件：
    {"type": "open", "width", "height", "fps", "frames", "classes"}
    {"type": "frame", "jpg": base64, "w", "h", "boxes": [[x, y, w, h, 类别序号, 分数]], "ms", "fps", "pos"}
    {"type": "ended"} / {"type": "error", "msg"}
标准输入命令：
    {"cmd": "conf", "value": 0.4}         调整置信度
    {"cmd": "snapshot", "path": "..."}    保存当前原始帧（原分辨率）
    {"cmd": "stop"}
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

# RTSP 默认走 TCP，丢包少、花屏少
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import torch  # noqa: E402

from . import models  # noqa: E402
from .protocol import emit, log  # noqa: E402


def is_stream(src: str) -> bool:
    return "://" in src


class Grabber(threading.Thread):
    """后台读帧，只保留最新一帧。视频文件按原始帧率播放，RTSP 断线自动重连。"""

    def __init__(self, src: str):
        super().__init__(daemon=True)
        self.src = src
        self.stream = is_stream(src)
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.pos = 0
        self.ended = False
        self.running = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not (1 <= self.fps <= 240):
            self.fps = 25.0

    def opened(self) -> bool:
        return self.cap.isOpened()

    def run(self):
        interval = 1.0 / self.fps
        fails = 0
        next_t = time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if not self.stream:
                    self.ended = True
                    return
                fails += 1
                if fails > 10:
                    log(f"视频流读取失败，第 {fails - 10} 次重连…")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                time.sleep(0.05)
                continue
            fails = 0
            with self.lock:
                self.frame = frame
                self.seq += 1
                self.pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if not self.stream:
                # 视频文件按原始速度播放
                next_t += interval
                delay = next_t - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.time()

    def latest(self):
        with self.lock:
            return (None if self.frame is None else self.frame, self.seq, self.pos)


def write_image(path: str, frame) -> None:
    # cv2.imwrite 不支持中文路径，先编码再写文件
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("图片编码失败")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buf.tobytes())


def run(cfg: dict) -> None:
    src = str(cfg["source"]).strip()
    conf = float(cfg.get("conf", 0.4))
    max_fps = float(cfg.get("max_fps", 12))
    preview_w = int(cfg.get("preview_width", 960))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = models.load(Path(cfg["weights"]), device)

    g = Grabber(src)
    if not g.opened():
        emit("error", msg=f"打不开视频源：{src}。RTSP 请检查地址、账号密码和网络；视频文件请检查路径和格式。")
        return
    w = int(g.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(g.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(g.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not g.stream else 0
    emit("open", width=w, height=h, fps=g.fps, frames=frames, classes=ck["classes"], device=str(device))
    g.start()

    state = {"conf": conf, "stop": False, "snapshot": None}

    def read_commands():
        for line in sys.stdin:
            try:
                c = json.loads(line.strip().lstrip("﻿"))
            except Exception:
                continue
            if c.get("cmd") == "conf":
                state["conf"] = float(c["value"])
            elif c.get("cmd") == "snapshot":
                state["snapshot"] = c["path"]
            elif c.get("cmd") == "stop":
                break
        state["stop"] = True  # 标准输入关闭也停止

    threading.Thread(target=read_commands, daemon=True).start()

    last_seq = -1
    t_prev = time.time()
    fps_avg = 0.0
    min_gap = 1.0 / max_fps if max_fps > 0 else 0
    with torch.no_grad():
        while not state["stop"]:
            frame, seq, pos = g.latest()
            if state["snapshot"] and frame is not None:
                path, state["snapshot"] = state["snapshot"], None
                try:
                    write_image(path, frame)
                    emit("snapshot", path=path)
                except Exception as e:
                    emit("snapshot", path=path, error=str(e))
            if frame is None or seq == last_seq:
                if g.ended:
                    emit("ended")
                    break
                time.sleep(0.01)
                continue
            last_seq = seq
            t0 = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().div_(255)
            out = model([x])[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            boxes = []
            for b, s, l in zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()):
                if s >= state["conf"]:
                    boxes.append([round(b[0], 1), round(b[1], 1), round(b[2] - b[0], 1), round(b[3] - b[1], 1), l - 1, round(s, 3)])
            fh, fw = frame.shape[:2]
            scale = min(1.0, preview_w / fw)
            small = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            now = time.time()
            fps_avg = 0.8 * fps_avg + 0.2 * (1 / max(now - t_prev, 1e-6)) if fps_avg else 1 / max(now - t_prev, 1e-6)
            t_prev = now
            emit(
                "frame",
                jpg=base64.b64encode(buf.tobytes()).decode("ascii") if ok else "",
                w=fw,
                h=fh,
                boxes=boxes,
                ms=round(ms, 1),
                fps=round(fps_avg, 1),
                pos=pos,
            )
            spare = min_gap - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)
    g.running = False
