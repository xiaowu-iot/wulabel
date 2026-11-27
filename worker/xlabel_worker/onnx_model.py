"""ONNX 模型（导入的外部模型、Ultralytics 等引擎训练后导出的模型）：用于预标注、体检、评估、误报测试、实时验证。

解码在 onnx_decode.py（检测、分割、关键点、分类），这里只负责界面提示文字和 PIL 图片的适配。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import onnx_decode
from .i18n import tr

KINDS = onnx_decode.KINDS

_NOTES = {
    "wulabel": "Wulabel 导出的模型",
    "e2e": "端到端输出（模型里已做过 NMS）",
    "no_names": "没有类别名，请手动填写",
}


def inspect(path: str) -> dict:
    """读模型的输入输出和元数据，猜格式、任务、类别名、类别数、输入尺寸"""
    info = onnx_decode.inspect_session(onnx_decode.session(path, use_gpu=False))
    note = info.get("note")
    info["note"] = tr(_NOTES[note]) if note in _NOTES else ""
    return info


class OnnxDetector:
    """兼容旧接口：predict(PIL 图, conf) → [(x0, y0, x1, y1, 类别序号, 分数)]；predict_full 返回完整结果"""

    def __init__(self, model: onnx_decode.OnnxModel):
        self.m = model

    @property
    def task(self) -> str:
        return self.m.task

    @property
    def classes(self) -> list[str]:
        return self.m.classes

    def predict_full(self, img: Image.Image, conf: float) -> list[dict]:
        return self.m.predict(np.asarray(img.convert("RGB")), conf)

    def predict(self, img: Image.Image, conf: float) -> list:
        return [(*d["box"], d["cls"], d["score"]) for d in self.predict_full(img, conf) if "box" in d]


def load(path: Path) -> tuple[OnnxDetector, dict]:
    """加载 model.onnx，旁边的 model.json 记着格式、任务和类别"""
    m, info = onnx_decode.load(path)
    return OnnxDetector(m), info
