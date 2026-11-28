"""推理服务包的文件夹模式：新放进来的图片、视频自动识别，结果按子文件夹（单号）放，表格追加一行，
识别过的不再识别，还在复制（刚改过）的先不识别，Wulabel 批量识别的结果文件夹跳过。

用假的检测器（不需要模型）。运行：python -m unittest discover -s worker/tests
"""

import collections
import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

SERVICE = Path(__file__).resolve().parents[1] / "service" / "service.py"


def load_service():
    spec = importlib.util.spec_from_file_location("wulabel_service_watch", SERVICE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeDet:
    """红色多的画面认成「扇叶损坏」0.9，其他什么都没有"""

    task = "detect"
    classes = ["扇叶损坏"]

    def detect(self, bgr, min_score=0.05):
        if bgr[..., 2].mean() > 150:
            h, w = bgr.shape[:2]
            return [{"label": "扇叶损坏", "score": 0.9, "box": [1, 1, w / 2, h / 2]}]
        return []


class FakeNotifier:
    def __init__(self):
        self.pushed = []

    def push(self, alert, image):
        self.pushed.append(alert)


def write_jpg(path: Path, bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", np.full((60, 80, 3), bgr, np.uint8))
    path.write_bytes(buf.tobytes())


def write_video(path: Path, bgr, seconds=3, fps=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (80, 60))
    for _ in range(seconds * fps):
        vw.write(np.full((60, 80, 3), bgr, np.uint8))
    vw.release()


class WatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = load_service()

    def test_watch_folder(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "in"
            write_jpg(src / "SF1" / "a.jpg", (0, 0, 220))  # 红
            write_jpg(src / "SF2" / "b.jpg", (200, 200, 200))  # 灰，没有目标
            write_video(src / "SF3" / "v.avi", (0, 0, 230))
            # Wulabel 批量识别的结果文件夹：跳过
            write_jpg(src / "结果" / "c.jpg", (0, 0, 220))
            (src / "结果" / ".wulabel-result").write_text("x")
            # 文件放好已经超过几秒
            old = time.time() - 60
            for f in src.rglob("*"):
                if f.is_file():
                    os.utime(f, (old, old))
            notifier = FakeNotifier()
            w = self.svc.FolderWatcher(
                {"dir": str(src), "out": str(d / "out"), "every": 1.0, "settle": 5},
                {"alert": {"min_score": 0.5}},
                FakeDet(),
                notifier,
                collections.deque(maxlen=10),
            )
            # 第一次扫描只记下大小，第二次没变才识别（还在复制的文件不识别）
            w.scan()
            self.assertEqual(w.status["processed"], 0)
            w.scan()
            self.assertEqual(w.status["processed"], 3)
            out = d / "out"
            rows = (out / "识别结果.csv").read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(rows), 4, rows)
            self.assertTrue(any(",SF2/b.jpg," in r and "没有目标" in r for r in rows), rows)
            # 截图按单号放
            self.assertTrue(list((out / "SF1").glob("a_*.jpg")))
            self.assertTrue(list((out / "SF3").glob("v_*.jpg")))
            self.assertFalse((out / "SF2").exists())
            # 红的两个推送了告警
            self.assertEqual(sorted(a["file"] for a in notifier.pushed), ["SF1/a.jpg", "SF3/v.avi"])
            # 再扫一遍：识别过的不再识别；改过的重新识别
            w.scan()
            self.assertEqual(w.status["processed"], 3)
            write_jpg(src / "SF2" / "b.jpg", (0, 0, 220))
            os.utime(src / "SF2" / "b.jpg", (old + 1, old + 1))
            w.scan()
            w.scan()
            self.assertEqual(w.status["processed"], 4)
            # 重启服务（新建一个）也记得识别过哪些
            w2 = self.svc.FolderWatcher({"dir": str(src), "out": str(out)}, {"alert": {}}, FakeDet(), FakeNotifier(), collections.deque())
            w2.scan()
            w2.scan()
            self.assertEqual(w2.status["processed"], 0)


if __name__ == "__main__":
    unittest.main()
