"""实时验证（live.infer）和推理服务包（service.py）对检测、分割、关键点、分类四类模型都能推理、画图。

用随机初始化的模型导出 ONNX 跑一遍；随机模型不一定出目标，所以另用假的推理输出检查分割轮廓、关键点、分类的字段。
运行：python -m unittest discover -s worker/tests（需要 torch、torchvision、onnx、onnxruntime、opencv，不下载权重）
"""

import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

SERVICE = Path(__file__).resolve().parents[1] / "service" / "service.py"


def load_service():
    spec = importlib.util.spec_from_file_location("wulabel_service", SERVICE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASES = [
    # 模型、任务、额外参数
    ("frcnn_mobilenet", "detect", {}),
    ("maskrcnn_r50", "segment", {}),
    ("keypointrcnn_r50", "pose", {"num_keypoints": 5}),
    ("cls_mobilenet", "classify", {}),
]


class DeployTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = load_service()

    def test_all_tasks(self):
        import torch

        from xlabel_worker import export, live, models

        svc = self.svc
        rng = np.random.default_rng(0)
        bgr = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        for key, task, extra in CASES:
            with self.subTest(model=key), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                size = 224 if task == "classify" else 160
                m = models.build(key, 2, size, pretrained=False, num_keypoints=int(extra.get("num_keypoints", 17)))
                models.save(tmp / "best.pt", m, key, ["甲", "乙"], size, extra)

                # 实时验证：一帧推理，结果按任务给出
                model, ck = models.load(tmp / "best.pt", torch.device("cpu"))
                with torch.no_grad():
                    boxes, polys, kpts, tags = live.infer(model, ck, models.task_of(key), bgr[:, :, ::-1].copy(), torch.device("cpu"), 0.0)
                if task == "classify":
                    self.assertEqual(boxes, [])
                    self.assertEqual(len(tags), 2)
                    self.assertAlmostEqual(sum(p for _, p in tags), 1.0, places=2)
                else:
                    self.assertEqual(tags, [])
                    self.assertEqual(len(polys), len(boxes) if task == "segment" else 0)
                    self.assertEqual(len(kpts), len(boxes) if task == "pose" else 0)
                    for k in kpts:
                        self.assertEqual(len(k), 5 * 3)

                # 推理服务包：导出 ONNX 后按输出自动认出任务，能推理、能画告警截图
                export.run(str(tmp / "best.pt"), str(tmp / "model.onnx"))
                det = svc.Detector(tmp / "model.onnx", False, 320)
                self.assertEqual(det.task, task)
                dets = det.detect(bgr, 0.0)
                for d in dets:
                    self.assertIn(d["label"], ["甲", "乙"])
                    self.assertEqual("box" in d, task != "classify")
                img = svc.draw(bgr, dets, [[0, 1], [1, 2]])
                self.assertEqual(img.shape, bgr.shape)

    def test_service_fields(self):
        """假的推理输出：分割给出轮廓、关键点给出点、分类没有框且不按区域过滤"""
        svc = self.svc
        bgr = np.zeros((240, 320, 3), dtype=np.uint8)

        class Fake:
            def __init__(self, outputs):
                self.outputs = outputs

            def run(self, _names, _feed):
                return self.outputs

        det = svc.Detector.__new__(svc.Detector)
        det.classes = ["甲", "乙"]
        det.max_side = 320
        det.size = 224
        det.input = "image"
        det.lock = threading.Lock()

        mask = np.zeros((1, 1, 240, 320), dtype=np.float32)
        mask[0, 0, 20:80, 30:120] = 0.9
        base = [np.array([[30, 20, 120, 80]], np.float32), np.array([2]), np.array([0.8], np.float32)]

        det.task, det.outputs = "segment", ["boxes", "labels", "scores", "masks"]
        det.sess = Fake(base + [mask])
        d = det.detect(bgr, 0.5)[0]
        self.assertEqual(d["label"], "乙")
        xs, ys = d["polygon"][0::2], d["polygon"][1::2]
        self.assertGreaterEqual(len(xs), 4)
        self.assertTrue(28 <= min(xs) and max(xs) <= 121 and 18 <= min(ys) and max(ys) <= 81)

        det.task, det.outputs = "pose", ["boxes", "labels", "scores", "keypoints", "keypoints_scores"]
        kp = np.array([[[40, 30, 1], [60, 50, 1], [80, 70, 1]]], np.float32)
        det.sess = Fake(base + [kp, np.array([[3.0, -1.0, 2.0]], np.float32)])
        d = det.detect(bgr, 0.5)[0]
        self.assertEqual(d["keypoints"], [40.0, 30.0, 2, 0, 0, 0, 80.0, 70.0, 2])

        det.task, det.outputs = "classify", ["probs"]
        det.sess = Fake([np.array([[0.2, 0.8]], np.float32)])
        dets = det.detect(bgr, 0.5)
        self.assertEqual(dets, [{"label": "乙", "score": 0.8}])
        self.assertTrue(svc.in_region(dets[0].get("box"), [[0, 0], [0.1, 0], [0.1, 0.1]], 320, 240))
        self.assertEqual(svc.draw(bgr, dets).shape, bgr.shape)

        # 多标签分类：分数够的类别都返回
        det.multi = True
        det.sess = Fake([np.array([[0.6, 0.7]], np.float32)])
        self.assertEqual([d["label"] for d in det.detect(bgr, 0.5)], ["乙", "甲"])
        det.multi = False


if __name__ == "__main__":
    unittest.main()
