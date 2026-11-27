"""通用 ONNX 解码（onnx_decode）：Ultralytics 系的检测、分割、关键点、分类、端到端输出都能解出来。

前半部分用假的推理输出（不需要真模型）；装了 ultralytics 时，后半部分用随机初始化的 YOLOv8 / YOLO11 真导出一遍再解码。
运行：python -m unittest discover -s worker/tests
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xlabel_worker import onnx_decode
from xlabel_worker.onnx_decode import OnnxModel


class Fake:
    def __init__(self, outputs):
        self.outputs = outputs

    def run(self, _names, _feed):
        return self.outputs


def fake_model(kind, task, outputs, size=64, classes=("a", "b"), kpt=None):
    m = OnnxModel.__new__(OnnxModel)
    m.kind, m.task, m.size, m.classes = kind, task, size, list(classes)
    m.kpt_shape, m.multi, m.max_side, m.no_batch = kpt, False, None, False
    m.input, m.out_names, m.sess = "images", [f"output{i}" for i in range(len(outputs))], Fake(outputs)
    return m


class FakeOutputTest(unittest.TestCase):
    # 128×64 的图：缩放比例 0.5，上下各补 16 像素（S = 64）
    img = np.zeros((64, 128, 3), dtype=np.uint8)

    def test_nms(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], np.float32)
        keep = onnx_decode.nms(boxes, np.array([0.9, 0.8, 0.7, 0.6]), np.array([0, 0, 1, 0]))
        self.assertEqual(sorted(keep.tolist()), [0, 2, 3])

    def test_yolov8_detect(self):
        o = np.zeros((1, 6, 3), np.float32)  # [4+C, N]
        o[0, :, 0] = [32, 32, 20, 10, 0.9, 0.1]  # 中心在画面中间，类别 a
        o[0, :, 1] = [33, 32, 20, 10, 0.8, 0.0]  # 和第一个重叠，去重后去掉
        o[0, :, 2] = [10, 30, 8, 8, 0.05, 0.6]  # 类别 b
        res = fake_model("yolov8", "detect", [o]).predict(self.img, 0.25)
        self.assertEqual([d["cls"] for d in res], [0, 1])
        x0, y0, x1, y1 = res[0]["box"]
        # 输入坐标 (22, 27)~(42, 37)，去掉上边 16 的补边、除以 0.5
        self.assertAlmostEqual(x0, 44, places=3)
        self.assertAlmostEqual(y0, 22, places=3)
        self.assertAlmostEqual(x1, 84, places=3)
        self.assertAlmostEqual(y1, 42, places=3)

    def test_yolov8_segment(self):
        nm, mh = 4, 16
        o = np.zeros((1, 4 + 2 + nm, 2), np.float32)
        o[0, :6, 0] = [32, 32, 32, 16, 0.9, 0.0]
        o[0, 6:, 0] = [10, 0, 0, 0]  # 只用第一个原型
        proto = np.full((1, nm, mh, mh), -10, np.float32)
        proto[0, 0, 6:10, 4:12] = 1  # 原型尺度上 (4,6)~(12,10)，对应输入 (16,24)~(48,40)
        d = fake_model("yolov8", "segment", [o, proto]).predict(self.img, 0.25)[0]
        poly = d["polygon"]
        self.assertGreaterEqual(len(poly), 6)
        xs, ys = poly[0::2], poly[1::2]
        # 原图上：x 32~96，y 16~48
        self.assertTrue(30 <= min(xs) and max(xs) <= 98, (min(xs), max(xs)))
        self.assertTrue(14 <= min(ys) and max(ys) <= 50, (min(ys), max(ys)))

    def test_yolov8_pose(self):
        k = 3
        o = np.zeros((1, 4 + 1 + k * 3, 1), np.float32)
        o[0, :5, 0] = [32, 32, 20, 20, 0.9]
        o[0, 5:, 0] = [30, 30, 0.9, 40, 36, 0.2, 34, 28, 0.8]
        d = fake_model("yolov8", "pose", [o], classes=["person"], kpt=[k, 3]).predict(self.img, 0.25)[0]
        self.assertEqual(d["cls"], 0)
        kp = d["keypoints"]
        self.assertEqual(len(kp), k * 3)
        self.assertAlmostEqual(kp[0], 60, places=3)  # (30 - 0) / 0.5
        self.assertAlmostEqual(kp[1], 28, places=3)  # (30 - 16) / 0.5
        self.assertEqual(kp[2], 2)
        self.assertEqual(kp[3:6], [0.0, 0.0, 0])  # 可见度低当成没找到

    def test_e2e(self):
        o = np.zeros((1, 300, 6), np.float32)
        o[0, 0] = [22, 27, 42, 37, 0.9, 1]
        o[0, 1] = [0, 0, 5, 5, 0.1, 0]
        res = fake_model("yolo_e2e", "detect", [o]).predict(self.img, 0.25)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["cls"], 1)
        self.assertAlmostEqual(res[0]["box"][0], 44, places=3)

    def test_classify(self):
        m = fake_model("yolov8", "classify", [np.array([[0.1, 0.7, 0.2]], np.float32)], classes=["x", "y", "z"])
        res = m.predict(self.img, 0.15)
        self.assertEqual([d["cls"] for d in res], [1, 2])


try:
    import ultralytics  # noqa: F401

    HAS_ULTRA = True
except Exception:
    HAS_ULTRA = False


@unittest.skipUnless(HAS_ULTRA, "没有安装 ultralytics")
class UltralyticsExportTest(unittest.TestCase):
    """随机初始化的模型（从结构文件建，不下载权重）导出 ONNX，检查格式、任务识别和解码不报错"""

    CASES = [
        ("yolov8n.yaml", "detect", "yolov8"),
        ("yolo11n-seg.yaml", "segment", "yolov8"),
        ("yolov8n-pose.yaml", "pose", "yolov8"),
        ("yolov8n-cls.yaml", "classify", "yolov8"),
        ("yolov10n.yaml", "detect", "yolo_e2e"),
    ]

    def test_export_and_decode(self):
        from ultralytics import YOLO

        img = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
        for cfg, task, kind in self.CASES:
            with self.subTest(model=cfg), tempfile.TemporaryDirectory() as tmp:
                size = 64
                cwd = os.getcwd()
                os.chdir(tmp)  # 从结构文件建的模型导出到当前目录
                try:
                    model = YOLO(cfg, task=task)
                    path = Path(model.export(format="onnx", imgsz=size, simplify=False, verbose=False)).resolve()
                finally:
                    os.chdir(cwd)
                info = onnx_decode.inspect_session(onnx_decode.session(path, use_gpu=False))
                self.assertEqual(info["task"], task)
                self.assertEqual(info["kind"], kind)
                self.assertTrue(info["classes"])
                if task == "pose":
                    self.assertEqual(info["kptShape"], [17, 3])
                (path.with_name("model.json")).write_text(json.dumps({"classes": info["classes"]}), encoding="utf-8")
                m, _ = onnx_decode.load(path, use_gpu=False)
                res = m.predict(img, 0.0)
                self.assertIsInstance(res, list)
                for d in res[:20]:
                    self.assertIn("cls", d)
                    if task == "classify":
                        self.assertNotIn("box", d)
                    else:
                        x0, y0, x1, y1 = d["box"]
                        self.assertTrue(0 <= x0 <= x1 <= 200 and 0 <= y0 <= y1 <= 120)
                    if task == "pose":
                        self.assertEqual(len(d["keypoints"]), 17 * 3)


if __name__ == "__main__":
    unittest.main()
