"""视频标注：从一帧的框出发，往后逐帧跟踪。

配置（JSON）：
    frames   ["第 0 帧（起始帧）路径", "第 1 帧", ...]
    boxes    [[x, y, w, h], ...]   起始帧上的框，原图像素坐标
    min_score  跟丢的阈值（默认 0.45，模板匹配的相关系数）
输出事件：
    frame {i, boxes: [[x, y, w, h] 或 null, ...]}   第 i 帧（从 1 开始）每个目标的位置，null 表示已跟丢
    done {frames, lost}
stdin 收到 {"cmd": "stop"} 时停下。

装了 opencv-contrib 时用 CSRT，否则用自带的模板匹配（搜索窗 + 多尺度 + 模板慢慢更新），
短距离跟踪够用；目标被挡住、大幅变形时会跟丢，跟丢就停，不乱跳。
"""

from __future__ import annotations

import json
import sys
import threading

import cv2
import numpy as np

from .protocol import emit, log

SCALES = (0.94, 1.0, 1.06)


def _read(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _csrt():
    for mod in (cv2, getattr(cv2, "legacy", None)):
        f = getattr(mod, "TrackerCSRT_create", None) if mod is not None else None
        if f is not None:
            return f
    return None


class TemplateTracker:
    """灰度模板匹配：在上一位置附近找最像的地方"""

    def __init__(self, gray: np.ndarray, box, min_score: float):
        x, y, w, h = box
        self.x, self.y, self.w, self.h = float(x), float(y), max(4.0, float(w)), max(4.0, float(h))
        self.min_score = min_score
        self.tpl = self._crop(gray, self.x, self.y, self.w, self.h)
        self.vx = self.vy = 0.0

    @staticmethod
    def _crop(gray, x, y, w, h):
        H, W = gray.shape[:2]
        x0, y0 = int(max(0, round(x))), int(max(0, round(y)))
        x1, y1 = int(min(W, round(x + w))), int(min(H, round(y + h)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return gray[y0:y1, x0:x1].astype(np.float32)

    def update(self, gray: np.ndarray):
        if self.tpl is None:
            return None
        H, W = gray.shape[:2]
        best = (-1.0, 0.0, 0.0, 1.0)
        # 预测位置（按上一帧的速度），搜索窗是目标大小的 2 倍左右
        px, py = self.x + self.vx, self.y + self.vy
        for s in SCALES:
            tw, th = max(4, int(round(self.tpl.shape[1] * s))), max(4, int(round(self.tpl.shape[0] * s)))
            tpl = cv2.resize(self.tpl, (tw, th)) if s != 1.0 else self.tpl
            mx, my = max(tw, 24), max(th, 24)
            sx0, sy0 = int(max(0, px - mx)), int(max(0, py - my))
            sx1, sy1 = int(min(W, px + self.w + mx)), int(min(H, py + self.h + my))
            if sx1 - sx0 <= tw or sy1 - sy0 <= th:
                continue
            region = gray[sy0:sy1, sx0:sx1].astype(np.float32)
            res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score > best[0]:
                best = (float(score), float(sx0 + loc[0]), float(sy0 + loc[1]), s)
        score, nx, ny, s = best
        if score < self.min_score:
            return None
        nw, nh = self.w * s, self.h * s
        self.vx, self.vy = 0.5 * self.vx + 0.5 * (nx - self.x), 0.5 * self.vy + 0.5 * (ny - self.y)
        self.x, self.y, self.w, self.h = nx, ny, nw, nh
        # 模板慢慢更新，适应外观变化；匹配得很好时才更新，免得把背景学进去
        cur = self._crop(gray, nx, ny, nw, nh)
        if cur is not None and score > 0.7:
            cur = cv2.resize(cur, (self.tpl.shape[1], self.tpl.shape[0]))
            self.tpl = 0.85 * self.tpl + 0.15 * cur
        return [self.x, self.y, self.w, self.h]


class CsrtTracker:
    def __init__(self, create, img: np.ndarray, box):
        self.t = create()
        x, y, w, h = box
        self.t.init(img, (int(round(x)), int(round(y)), max(4, int(round(w))), max(4, int(round(h)))))

    def update(self, img: np.ndarray):
        ok, b = self.t.update(img)
        if not ok:
            return None
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]


def _clip(b, W, H):
    x, y, w, h = b
    x0, y0 = min(max(0.0, x), W - 1.0), min(max(0.0, y), H - 1.0)
    x1, y1 = min(max(x0 + 1.0, x + w), float(W)), min(max(y0 + 1.0, y + h), float(H))
    # 大半出了画面就当跟丢
    if (x1 - x0) * (y1 - y0) < 0.4 * w * h:
        return None
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


def run(cfg: dict) -> None:
    frames: list[str] = cfg["frames"]
    boxes = cfg["boxes"]
    min_score = float(cfg.get("min_score", 0.45))
    stop = threading.Event()

    def watch():
        for line in sys.stdin:
            try:
                if json.loads(line).get("cmd") == "stop":
                    stop.set()
                    return
            except ValueError:
                pass

    threading.Thread(target=watch, daemon=True).start()

    first = _read(frames[0])
    if first is None:
        raise RuntimeError(f"读不了起始帧：{frames[0]}")
    create = _csrt()
    log("跟踪方法：" + ("CSRT" if create else "模板匹配"))
    gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    trackers = [CsrtTracker(create, first, b) if create else TemplateTracker(gray0, b, min_score) for b in boxes]
    alive = [True] * len(boxes)
    done = 0
    for i, path in enumerate(frames[1:], start=1):
        if stop.is_set() or not any(alive):
            break
        img = _read(path)
        if img is None:
            break
        H, W = img.shape[:2]
        src = img if create else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = []
        for k, t in enumerate(trackers):
            b = t.update(src) if alive[k] else None
            b = _clip(b, W, H) if b is not None else None
            if b is None:
                alive[k] = False
            out.append(b)
        emit("frame", i=i, boxes=out)
        done = i
    emit("done", frames=done, lost=sum(1 for a in alive if not a))
