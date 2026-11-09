"""xlabel 推理服务：多路视频流检测 + 告警推送 + HTTP 接口。

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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

HERE = Path(__file__).resolve().parent
log = logging.getLogger("xlabel")


# ---------------- 检测 ----------------


class Detector:
    def __init__(self, model_path: Path, use_gpu: bool, max_side: int):
        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.sess = ort.InferenceSession(str(model_path), providers=providers)
        self.input = self.sess.get_inputs()[0].name
        meta = json.loads(model_path.with_name("model.json").read_text(encoding="utf-8"))
        self.classes: list[str] = meta["classes"]
        self.max_side = max_side
        self.lock = threading.Lock()
        log.info("模型已加载：%s，类别 %s，推理设备 %s", model_path.name, self.classes, self.sess.get_providers()[0])

    def detect(self, bgr: np.ndarray, min_score: float = 0.05) -> list[dict]:
        h, w = bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        img = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        with self.lock:
            boxes, labels, scores = self.sess.run(None, {self.input: x})
        out = []
        for b, l, s in zip(boxes, labels, scores):
            if s < min_score:
                continue
            x0, y0, x1, y1 = (b / scale).tolist()
            cls = int(l) - 1
            out.append(
                {
                    "label": self.classes[cls] if 0 <= cls < len(self.classes) else str(cls),
                    "score": round(float(s), 4),
                    "box": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                }
            )
        return out


def draw(bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    img = bgr.copy()
    for d in dets:
        x, y, w, h = map(int, d["box"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, f"{d['label']} {d['score']:.2f}", (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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
            raise RuntimeError(f"MQTT 连接被拒绝：{ack!r}")
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
            log.warning("告警推送队列已满，丢弃一条告警")

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
                    log.warning("Webhook 推送失败：%s", e)
            mq = self.cfg.get("mqtt") or {}
            if mq.get("host"):
                try:
                    mqtt_publish(mq, json.dumps(alert, ensure_ascii=False).encode("utf-8"))
                except Exception as e:
                    log.warning("MQTT 推送失败：%s", e)


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


def in_region(box: list[float], region: list[list[float]] | None, w: int, h: int) -> bool:
    """区域是归一化坐标（0~1）的多边形；判断框的底边中点（人、车站立的位置）"""
    if not region or len(region) < 3:
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
        is_file = "://" not in url
        while self.running:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.status["connected"] = False
                self.status["errors"] += 1
                log.warning("[%s] 连接失败，10 秒后重试", self.name)
                time.sleep(10)
                continue
            self.status["connected"] = True
            log.info("[%s] 已连接", self.name)
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
            log.warning("[%s] 断开，5 秒后重连", self.name)
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
                    log.exception("[%s] 检测出错", self.name)
            time.sleep(max(0.05, interval - (time.time() - t0)))

    def handle(self, frame: np.ndarray, rules: dict):
        now = dt.datetime.now()
        h, w = frame.shape[:2]
        dets = self.det.detect(frame, min_score=0.05)
        self.status["last_detect"] = time.time()
        min_score = float(rules.get("min_score", 0.5))
        labels = set(rules.get("labels") or [])
        region = self.s.get("region")
        hits = [
            d for d in dets
            if d["score"] >= min_score and (not labels or d["label"] in labels) and in_region(d["box"], region, w, h)
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
        img = draw(frame, hits)
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
        log.info("[%s] 告警：%s", self.name, "、".join(alert["labels"]))
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
                self.send_json({"ok": True, "classes": det.classes, "streams": [s.status for s in streams]})
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
                    self.send_json({"error": "文件不存在"}, 404)
            else:
                self.send_json({"error": "接口不存在", "接口": ["GET /health", "GET /alerts?limit=50", "POST /predict"]}, 404)

        def do_POST(self):
            if self.path.split("?")[0] != "/predict":
                self.send_json({"error": "接口不存在"}, 404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("Content-Type", "").startswith("application/json"):
                try:
                    body = base64.b64decode(json.loads(body)["image"])
                except Exception:
                    self.send_json({"error": "JSON 需要 image 字段（base64 图片）"}, 400)
                    return
            img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self.send_json({"error": "图片解码失败"}, 400)
                return
            q = dict(p.split("=", 1) for p in self.path.partition("?")[2].split("&") if "=" in p)
            t0 = time.time()
            dets = det.detect(img, float(q.get("conf", 0.4)))
            self.send_json({"width": img.shape[1], "height": img.shape[0], "detections": dets, "ms": round((time.time() - t0) * 1000, 1)})

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
    log.info("推理服务已启动：%d 路视频，HTTP 接口 http://%s:%d", len(streams), cfg.get("host", "0.0.0.0"), port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("已停止")


if __name__ == "__main__":
    main()
