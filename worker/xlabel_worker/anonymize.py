"""批量打码：把人脸、车牌等区域打马赛克 / 模糊 / 涂黑，直接改写给定的图片文件（只用于导出的副本、标注包里的副本）。

配置（JSON）：
    files    [{"path": "...", "regions": [[x, y, w, h], ...]}]   regions 是已知要打码的区域（比如标注里的人脸）
    auto     ["face", "license plate"]   另外自动找这些东西（开放词汇检测，需要 transformers）
    conf     自动检测的阈值（默认 0.15）
    mode     mosaic 马赛克 / blur 模糊 / black 涂黑
输出事件：progress {done, total}；done {files, regions}
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .protocol import emit, log

QUERIES = {"face": ["human face"], "license plate": ["license plate", "car number plate"]}


def _apply(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, mode: str) -> None:
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    if mode == "black":
        roi[:] = 0
    elif mode == "blur":
        k = max(3, (min(w, h) // 3) | 1)
        img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        # 马赛克：缩小再放大，每块大约是区域短边的 1/8
        block = max(4, min(w, h) // 8)
        small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
        img[y0:y1, x0:x1] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def run(cfg: dict) -> None:
    files = cfg["files"]
    mode = cfg.get("mode", "mosaic")
    auto = [a for a in cfg.get("auto") or [] if a in QUERIES]
    conf = float(cfg.get("conf", 0.15))
    owl = None
    queries: list[str] = []
    if auto:
        from .assist import Owl

        mps = getattr(torch.backends, "mps", None)
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if (mps is not None and mps.is_available()) else "cpu")
        owl = Owl(device)
        for a in auto:
            queries += QUERIES[a]
        log(f"自动查找：{'、'.join(auto)}")
    total = len(files)
    n_regions = 0
    for i, f in enumerate(files):
        path = Path(f["path"])
        # 用 imdecode / imencode，路径里有中文也没问题
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        hh, ww = img.shape[:2]
        boxes = [(x, y, x + w, y + h) for x, y, w, h in f.get("regions") or []]
        if owl is not None:
            boxes += [b for b, _, _ in owl.detect(str(path), queries, conf)]
        for x0, y0, x1, y1 in boxes:
            # 稍微放大一圈，免得边缘露出来
            px, py = (x1 - x0) * 0.08, (y1 - y0) * 0.08
            a0, b0 = max(0, int(x0 - px)), max(0, int(y0 - py))
            a1, b1 = min(ww, int(x1 + px + 1)), min(hh, int(y1 + py + 1))
            if a1 > a0 and b1 > b0:
                _apply(img, a0, b0, a1, b1, mode)
                n_regions += 1
        ext = path.suffix.lower() or ".jpg"
        ok, buf = cv2.imencode(ext if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff") else ".jpg", img)
        if ok:
            buf.tofile(str(path))
        if (i + 1) % 5 == 0 or i + 1 == total:
            emit("progress", done=i + 1, total=total)
    emit("done", files=total, regions=n_regions)
