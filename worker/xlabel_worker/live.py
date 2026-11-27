"""实时验证：读取视频流（RTSP、HTTP-FLV、HLS、http 上的 MP4 等）或视频文件，用模型逐帧推理。

配置（JSON）：source（视频流地址或视频文件路径）、weights、conf、max_fps、preview_width
输出事件：
    {"type": "stage", "step": "model" / "open"}   启动阶段：加载模型、连接视频源
    {"type": "open", "width", "height", "fps", "frames", "classes", "task"}
    {"type": "frame", "jpg": base64, "w", "h", "boxes": [[x, y, w, h, 类别序号, 分数]], "ms", "fps", "pos",
     "polys": 分割模型每个框的多边形 [x1, y1, ...]（和 boxes 一一对应，没有时为 []），
     "kpts": 关键点模型每个框的关键点 [x, y, v, ...]（和 boxes 一一对应），
     "tags": 分类模型前几名 [[类别序号, 概率]]}
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
from PIL import Image  # noqa: E402
from torchvision.transforms import functional as F  # noqa: E402

from . import models  # noqa: E402
from .assist import mask_to_poly  # noqa: E402
from .data import cls_preprocess  # noqa: E402
from .i18n import tr  # noqa: E402
from .protocol import emit, log  # noqa: E402


def is_stream(src: str) -> bool:
    return "://" in src


class Grabber(threading.Thread):
    """后台读帧，只保留最新一帧。视频文件和 http 点播（有总帧数）按原始帧率播放、放完结束；直播流断线自动重连。"""

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
        # http(s) 上的 MP4、点播 m3u8 有总帧数，当成视频文件播放；直播流没有总帧数
        if self.stream and src.lower().startswith(("http://", "https://")) and (self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0:
            self.stream = False

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
                    log(tr("视频流读取失败，第 {n} 次重连…", n=fails - 10))
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
        raise RuntimeError(tr("图片编码失败"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buf.tobytes())


def infer(model, ck: dict, task: str, rgb, device: torch.device, conf: float):
    """一帧推理，按任务返回 (boxes, polys, kpts, tags)，格式见文件开头的说明"""
    if task == "classify":
        x = cls_preprocess(Image.fromarray(rgb), int(ck["img_size"])).unsqueeze(0).to(device)
        if ck.get("multi_label"):
            # 多标签：分数够阈值的类别都算，从高到低
            prob = torch.sigmoid(model(x).float())[0].cpu()
            order = torch.argsort(prob, descending=True)
            return [], [], [], [[int(i), round(float(prob[i]), 3)] for i in order if float(prob[i]) >= conf]
        prob = torch.softmax(model(x).float(), dim=1)[0].cpu()
        top = torch.topk(prob, min(3, prob.numel()))
        return [], [], [], [[int(i), round(float(s), 3)] for s, i in zip(top.values, top.indices)]
    out = model([F.to_tensor(rgb).to(device)])[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    masks = out["masks"].cpu().numpy() if task == "segment" and "masks" in out else None
    kps = out["keypoints"].cpu().numpy() if task == "pose" and "keypoints" in out else None
    ksc = out["keypoints_scores"].cpu().numpy() if kps is not None and "keypoints_scores" in out else None
    boxes, polys, kpts = [], [], []
    for i, (b, s, l) in enumerate(zip(out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist())):
        if s < conf:
            continue
        boxes.append([round(b[0], 1), round(b[1], 1), round(b[2] - b[0], 1), round(b[3] - b[1], 1), l - 1, round(s, 3)])
        if masks is not None:
            polys.append(mask_to_poly(masks[i, 0]) or [])
        if kps is not None:
            flat = []
            for j, (kx, ky, _) in enumerate(kps[i]):
                # 关键点热图分数 > 0 当成找到了，和预标注一致
                ok = ksc is None or ksc[i, j] > 0
                flat += [round(float(kx), 1), round(float(ky), 1), 2] if ok else [0, 0, 0]
            kpts.append(flat)
    return boxes, polys, kpts, []


def infer_onnx(model, task: str, rgb, conf: float):
    """ONNX 模型（Ultralytics 等引擎训练后导出的、导入的）一帧推理，返回格式和 infer 一样"""
    res = model.m.predict(rgb, conf)
    if task == "classify":
        return [], [], [], [[d["cls"], round(d["score"], 3)] for d in res[: 3 if not model.m.multi else None]]
    boxes, polys, kpts = [], [], []
    for d in res:
        x0, y0, x1, y1 = d["box"]
        boxes.append([round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1), d["cls"], round(d["score"], 3)])
        if task == "segment":
            polys.append([round(v, 1) for v in d.get("polygon") or []])
        if task == "pose":
            kpts.append([round(v, 1) if j % 3 < 2 else int(v) for j, v in enumerate(d.get("keypoints") or [])])
    return boxes, polys, kpts, []


def run(cfg: dict) -> None:
    src = str(cfg["source"]).strip()
    conf = float(cfg.get("conf", 0.4))
    max_fps = float(cfg.get("max_fps", 12))
    preview_w = int(cfg.get("preview_width", 960))
    mps = getattr(torch.backends, "mps", None)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if (mps is not None and mps.is_available()) else "cpu"
    )
    # 启动阶段告诉界面在做什么：加载模型、连接视频源（连 RTSP 可能要好几秒）
    emit("stage", step="model")
    weights = Path(cfg["weights"])
    onnx = weights.suffix.lower() == ".onnx"
    if onnx:
        from . import onnx_model

        model, ck = onnx_model.load(weights)
        device = torch.device("cuda" if "CUDA" in model.m.provider else "cpu")
    else:
        model, ck = models.load(weights, device)

    emit("stage", step="open")
    g = Grabber(src)
    if not g.opened():
        emit("error", msg=tr("打不开视频源：{src}。视频流请检查地址、账号密码和网络；视频文件请检查路径和格式。", src=src))
        return
    w = int(g.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(g.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(g.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not g.stream else 0
    task = model.task if onnx else models.task_of(ck["model"])
    emit("open", width=w, height=h, fps=g.fps, frames=frames, classes=ck["classes"], device=str(device), task=task)
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
            if onnx:
                boxes, polys, kpts, tags = infer_onnx(model, task, rgb, state["conf"])
            else:
                boxes, polys, kpts, tags = infer(model, ck, task, rgb, device, state["conf"])
            ms = (time.time() - t0) * 1000
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
                polys=polys,
                kpts=kpts,
                tags=tags,
                ms=round(ms, 1),
                fps=round(fps_avg, 1),
                pos=pos,
            )
            spare = min_gap - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)
    g.running = False
