"""xlabel 推理服务：多路视频流推理（检测、分割、关键点、分类）+ 告警推送 + HTTP 接口。

视频流支持 RTSP、HTTP-FLV、HLS（m3u8）、http 上的 MP4，也可以是本地视频文件。

只依赖 onnxruntime、opencv-python-headless、numpy，配置见 config.json，说明见 README.md。
运行：python service.py [config.json]
"""

from __future__ import annotations

import base64
import collections
import datetime as dt
import json
import logging
import os
import queue
import socket
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# 提示文字的语言：服务包里带 locale.json（中文原文 → 译文）时用它，没有就是中文
try:
    _LOCALE = json.loads((Path(__file__).with_name("locale.json")).read_text(encoding="utf-8"))
except (OSError, ValueError):
    _LOCALE = {}


def tr(zh: str) -> str:
    return _LOCALE.get(zh) or zh


import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

HERE = Path(__file__).resolve().parent
log = logging.getLogger("xlabel")


# ---------------- 推理 ----------------


def mask_to_poly(mask: np.ndarray, scale: float) -> list[float]:
    """一个目标的掩膜概率图转成多边形 [x1, y1, x2, y2, ...]（原图像素），取面积最大的一块"""
    m = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return []
    pts = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(pts) < 3:
        return []
    return [round(float(v) / scale, 1) for p in pts for v in p]


class Detector:
    """按导出的输出自动区分任务：probs → 分类；masks → 实例分割；keypoints → 关键点；否则目标检测"""

    def __init__(self, model_path: Path, use_gpu: bool, max_side: int):
        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.sess = ort.InferenceSession(str(model_path), providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input = inp.name
        self.outputs = [o.name for o in self.sess.get_outputs()]
        self.task = (
            "classify" if "probs" in self.outputs
            else "segment" if "masks" in self.outputs
            else "pose" if "keypoints" in self.outputs
            else "detect"
        )
        # 分类模型输入固定尺寸 [N, 3, S, S]
        self.size = inp.shape[-1] if isinstance(inp.shape[-1], int) else 224
        meta = json.loads(model_path.with_name("model.json").read_text(encoding="utf-8"))
        self.classes: list[str] = meta["classes"]
        # 多标签分类：每个类别各自的分数，够阈值的都算
        self.multi = bool(meta.get("multiLabel"))
        self.max_side = max_side
        self.lock = threading.Lock()
        log.info(tr("模型已加载：%s，类别 %s，推理设备 %s"), model_path.name, self.classes, self.sess.get_providers()[0])

    def name_of(self, i: int) -> str:
        return self.classes[i] if 0 <= i < len(self.classes) else str(i)

    def detect(self, bgr: np.ndarray, min_score: float = 0.05) -> list[dict]:
        """检测、分割、关键点：每个目标一项，带 box [x, y, w, h]（分割多 polygon，关键点多 keypoints [x, y, v, ...]）。
        分类：概率不低于 min_score 的类别各一项（按概率从高到低，最多 5 个；多标签分类不限个数），没有 box"""
        if self.task == "classify":
            rgb = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            with self.lock:
                probs = self.sess.run(None, {self.input: x})[0][0]
            order = np.argsort(-probs) if getattr(self, "multi", False) else np.argsort(-probs)[:5]
            return [{"label": self.name_of(int(i)), "score": round(float(probs[i]), 4)} for i in order if probs[i] >= min_score]
        h, w = bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        img = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        with self.lock:
            res = dict(zip(self.outputs, self.sess.run(None, {self.input: x})))
        out = []
        for i, (b, l, s) in enumerate(zip(res["boxes"], res["labels"], res["scores"])):
            if s < min_score:
                continue
            x0, y0, x1, y1 = (b / scale).tolist()
            d = {
                "label": self.name_of(int(l) - 1),
                "score": round(float(s), 4),
                "box": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
            }
            if self.task == "segment":
                d["polygon"] = mask_to_poly(res["masks"][i, 0], scale)
            elif self.task == "pose":
                ks = res.get("keypoints_scores")
                flat = []
                for j, (kx, ky, _) in enumerate(res["keypoints"][i]):
                    # 关键点分数 > 0 当成找到了（可见性记 2），否则记 0
                    found = ks is None or ks[i, j] > 0
                    flat += [round(float(kx) / scale, 1), round(float(ky) / scale, 1), 2] if found else [0, 0, 0]
                d["keypoints"] = flat
            out.append(d)
        return out


def draw(bgr: np.ndarray, dets: list[dict], skeleton: list | None = None) -> np.ndarray:
    img = bgr.copy()
    red = (0, 0, 255)
    for k, d in enumerate(dets):
        if "box" not in d:
            # 分类：整张图的类别写在左上角
            cv2.putText(img, f"{d['label']} {d['score']:.2f}", (10, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2)
            continue
        x, y, w, h = map(int, d["box"])
        poly = d.get("polygon")
        if poly:
            cv2.polylines(img, [np.array(poly, dtype=np.int32).reshape(-1, 2)], True, red, 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), red, 2)
        kp = d.get("keypoints")
        if kp:
            pts = [(int(kp[i]), int(kp[i + 1])) if kp[i + 2] > 0 else None for i in range(0, len(kp), 3)]
            for a, b in skeleton or []:
                if 0 <= a < len(pts) and 0 <= b < len(pts) and pts[a] and pts[b]:
                    cv2.line(img, pts[a], pts[b], (0, 200, 255), 2)
            for p in pts:
                if p:
                    cv2.circle(img, p, 4, (0, 255, 0), -1)
        cv2.putText(img, f"{d['label']} {d['score']:.2f}", (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
    return img


def save_jpg(path: Path, img: np.ndarray, quality: int = 90) -> None:
    # cv2.imwrite 不支持中文路径
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.tobytes())


# ---------------- 推送 ----------------


def mqtt_publish(cfg: dict, payload: bytes) -> None:
    """最简 MQTT 3.1.1 发布（QoS 0），不依赖第三方库"""

    def s16(b: bytes) -> bytes:
        return struct.pack("!H", len(b)) + b

    def rlen(n: int) -> bytes:
        out = b""
        while True:
            d, n = n % 128, n // 128
            out += bytes([d | (0x80 if n else 0)])
            if not n:
                return out

    flags = 0x02
    body = s16(cfg.get("client_id", f"xlabel-{os.getpid()}").encode())
    if cfg.get("username"):
        flags |= 0x80
        body += s16(cfg["username"].encode())
    if cfg.get("password"):
        flags |= 0x40
        body += s16(cfg["password"].encode())
    var = s16(b"MQTT") + bytes([4, flags]) + struct.pack("!H", 30)
    with socket.create_connection((cfg["host"], int(cfg.get("port", 1883))), timeout=5) as s:
        s.sendall(b"\x10" + rlen(len(var) + len(body)) + var + body)
        ack = s.recv(4)
        if len(ack) < 4 or ack[0] != 0x20 or ack[3] != 0:
            raise RuntimeError(tr("MQTT 连接被拒绝：") + repr(ack))
        pub = s16(cfg["topic"].encode()) + payload
        s.sendall(b"\x30" + rlen(len(pub)) + pub)
        s.sendall(b"\xe0\x00")


class Notifier(threading.Thread):
    """告警推送放到单独线程，网络慢不影响检测"""

    def __init__(self, cfg: dict):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q: queue.Queue = queue.Queue(maxsize=200)

    def push(self, alert: dict, image: np.ndarray | None) -> None:
        try:
            self.q.put_nowait((alert, image))
        except queue.Full:
            log.warning(tr("告警推送队列已满，丢弃一条告警"))

    def run(self):
        while True:
            alert, image = self.q.get()
            msg = dict(alert)
            wh = self.cfg.get("webhook") or {}
            if wh.get("url"):
                if wh.get("with_image") and image is not None:
                    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        msg["image_base64"] = base64.b64encode(buf.tobytes()).decode()
                try:
                    req = urllib.request.Request(
                        wh["url"],
                        data=json.dumps(msg, ensure_ascii=False).encode("utf-8"),
                        headers={"Content-Type": "application/json", **(wh.get("headers") or {})},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=10).read()
                except Exception as e:
                    log.warning(tr("Webhook 推送失败：%s"), e)
            mq = self.cfg.get("mqtt") or {}
            if mq.get("host"):
                try:
                    mqtt_publish(mq, json.dumps(alert, ensure_ascii=False).encode("utf-8"))
                except Exception as e:
                    log.warning(tr("MQTT 推送失败：%s"), e)


# ---------------- 规则 ----------------


def in_hours(spec: str, now: dt.datetime) -> bool:
    """生效时段，如 "07:00-19:00"；跨天写 "22:00-06:00"；为空表示全天"""
    if not spec:
        return True
    try:
        a, b = [dt.datetime.strptime(x.strip(), "%H:%M").time() for x in spec.split("-")]
    except ValueError:
        return True
    t = now.time()
    return a <= t <= b if a <= b else (t >= a or t <= b)


def in_region(box: list[float] | None, region: list[list[float]] | None, w: int, h: int) -> bool:
    """区域是归一化坐标（0~1）的多边形；判断框的底边中点（人、车站立的位置）。分类结果没有框，不按区域过滤"""
    if not region or len(region) < 3 or box is None:
        return True
    poly = np.array([[p[0] * w, p[1] * h] for p in region], dtype=np.float32)
    x, y, bw, bh = box
    return cv2.pointPolygonTest(poly, (x + bw / 2, y + bh), False) >= 0


# ---------------- 视频流 ----------------


class Stream(threading.Thread):
    def __init__(self, scfg: dict, cfg: dict, det: Detector, notifier: Notifier, alerts: collections.deque):
        super().__init__(daemon=True)
        self.s = scfg
        self.cfg = cfg
        self.det = det
        self.notifier = notifier
        self.alerts = alerts
        self.name = scfg.get("name") or scfg["url"]
        self.status = {"name": self.name, "connected": False, "last_frame": None, "last_detect": None, "frames": 0, "errors": 0}
        self.frame = None
        self.lock = threading.Lock()
        self.cooldown: dict[str, float] = {}
        self.last_reflux = 0.0
        self.running = True

    def reader(self):
        url = self.s["url"]
        while self.running:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.status["connected"] = False
                self.status["errors"] += 1
                log.warning(tr("[%s] 连接失败，10 秒后重试"), self.name)
                time.sleep(10)
                continue
            # 本地视频文件和 http(s) 点播（MP4、点播 m3u8，有总帧数）按原速循环播放；直播流有多少读多少
            is_file = "://" not in url or (
                url.lower().startswith(("http://", "https://")) and (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
            )
            self.status["connected"] = True
            log.info(tr("[%s] 已连接"), self.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    if is_file and self.s.get("loop", True):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                with self.lock:
                    self.frame = frame
                self.status["frames"] += 1
                self.status["last_frame"] = time.time()
                if is_file:
                    time.sleep(1 / fps)
            cap.release()
            self.status["connected"] = False
            log.warning(tr("[%s] 断开，5 秒后重连"), self.name)
            time.sleep(5)

    def run(self):
        threading.Thread(target=self.reader, daemon=True).start()
        interval = float(self.s.get("interval", self.cfg.get("interval", 1.0)))
        rules = self.cfg.get("alert", {})
        while self.running:
            t0 = time.time()
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is not None:
                try:
                    self.handle(frame, rules)
                except Exception:
                    log.exception(tr("[%s] 检测出错"), self.name)
            time.sleep(max(0.05, interval - (time.time() - t0)))

    def handle(self, frame: np.ndarray, rules: dict):
        now = dt.datetime.now()
        h, w = frame.shape[:2]
        dets = self.det.detect(frame, min_score=0.05)
        self.status["last_detect"] = time.time()
        min_score = float(rules.get("min_score", 0.5))
        # 每个类别单独的阈值（多标签分类用评估时按类别算出的），没写的类别用 min_score
        class_scores = rules.get("class_scores") or {}
        labels = set(rules.get("labels") or [])
        region = self.s.get("region")
        hits = [
            d for d in dets
            if d["score"] >= float(class_scores.get(d["label"], min_score))
            and (not labels or d["label"] in labels)
            and in_region(d.get("box"), region, w, h)
        ]
        # 现场回流：拿不准的画面存下来，回到 xlabel 里标注
        rf = self.cfg.get("reflux", {})
        if rf.get("enabled", True):
            lo, hi = float(rf.get("low", 0.3)), float(rf.get("high", 0.5))
            if any(lo <= d["score"] < hi for d in dets) and time.time() - self.last_reflux >= float(rf.get("interval", 60)):
                self.last_reflux = time.time()
                save_jpg(HERE / "reflux" / now.strftime("%Y%m%d") / f"{safe(self.name)}_{now:%H%M%S}.jpg", frame, 95)
        if not hits or not in_hours(rules.get("hours", ""), now):
            return
        cool = float(rules.get("cooldown", 60))
        fresh = []
        for d in hits:
            last = self.cooldown.get(d["label"], 0)
            if time.time() - last >= cool:
                fresh.append(d)
        if not fresh:
            return
        for d in fresh:
            self.cooldown[d["label"]] = time.time()
        img = draw(frame, hits, self.cfg.get("skeleton"))
        snap = None
        if rules.get("save_image", True):
            snap = HERE / "alerts" / now.strftime("%Y%m%d") / f"{safe(self.name)}_{now:%H%M%S}.jpg"
            save_jpg(snap, img)
        alert = {
            "stream": self.name,
            "time": now.isoformat(timespec="seconds"),
            "labels": sorted({d["label"] for d in fresh}),
            "detections": hits,
            "width": w,
            "height": h,
            "image": str(snap.relative_to(HERE)).replace("\\", "/") if snap else None,
        }
        # 可以直接拼在服务地址后面访问的截图地址（已转义中文）
        alert["image_url"] = "/" + urllib.parse.quote(alert["image"]) if alert["image"] else None
        self.alerts.appendleft(alert)
        log.info(tr("[%s] 告警：%s"), self.name, tr("、").join(alert["labels"]))
        self.notifier.push(alert, img)


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]


# ---------------- HTTP 接口 ----------------


def make_handler(det: Detector, streams: list[Stream], alerts: collections.deque):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            # 路径里可能有中文（截图文件名带视频流名称）
            path = urllib.parse.unquote(self.path.split("?")[0])
            if path == "/health":
                self.send_json({"ok": True, "task": det.task, "classes": det.classes, "streams": [s.status for s in streams]})
            elif path == "/alerts":
                q = dict(p.split("=", 1) for p in self.path.partition("?")[2].split("&") if "=" in p)
                self.send_json(list(alerts)[: int(q.get("limit", 50))])
            elif path.startswith("/alerts/") or path.startswith("/reflux/"):
                f = (HERE / path.lstrip("/")).resolve()
                if HERE in f.parents and f.is_file():
                    data = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_json({"error": tr("文件不存在")}, 404)
            else:
                self.send_json({"error": tr("接口不存在"), "endpoints": ["GET /health", "GET /alerts?limit=50", "POST /predict"]}, 404)

        def do_POST(self):
            if self.path.split("?")[0] != "/predict":
                self.send_json({"error": tr("接口不存在")}, 404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("Content-Type", "").startswith("application/json"):
                try:
                    body = base64.b64decode(json.loads(body)["image"])
                except Exception:
                    self.send_json({"error": tr("JSON 需要 image 字段（base64 图片）")}, 400)
                    return
            img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self.send_json({"error": tr("图片解码失败")}, 400)
                return
            q = dict(p.split("=", 1) for p in self.path.partition("?")[2].split("&") if "=" in p)
            t0 = time.time()
            dets = det.detect(img, float(q.get("conf", 0.4)))
            self.send_json(
                {"task": det.task, "width": img.shape[1], "height": img.shape[0], "detections": dets, "ms": round((time.time() - t0) * 1000, 1)}
            )

    return Handler


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    (HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(HERE / "logs" / "service.log", encoding="utf-8")],
    )
    det = Detector(HERE / cfg.get("model", "model.onnx"), bool(cfg.get("gpu", True)), int(cfg.get("max_side", 1280)))
    notifier = Notifier(cfg)
    notifier.start()
    alerts: collections.deque = collections.deque(maxlen=500)
    streams = [Stream(s, cfg, det, notifier, alerts) for s in cfg.get("streams", []) if s.get("url")]
    for s in streams:
        s.start()
    port = int(cfg.get("port", 8090))
    server = ThreadingHTTPServer((cfg.get("host", "0.0.0.0"), port), make_handler(det, streams, alerts))
    log.info(tr("推理服务已启动：%d 路视频，HTTP 接口 http://%s:%d"), len(streams), cfg.get("host", "0.0.0.0"), port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info(tr("已停止"))


if __name__ == "__main__":
    main()
