"""Ultralytics 训练引擎冒烟测试：小数据集上 CPU 从结构文件训 1 轮 → 自动导出 ONNX → Wulabel 评估 → 用 ONNX 预测；
再接着训一次、停止后从断点继续。没装 ultralytics 时跳过。

运行：python -m unittest discover -s worker/tests
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_yolox_smoke import CLASSES, make_dataset

try:
    import ultralytics  # noqa: F401

    HAS_ULTRA = True
except Exception:
    HAS_ULTRA = False


def coco_to_yolo(src: Path, dst: Path) -> None:
    """把测试用的 COCO 数据集换成桌面端导出的 YOLO 格式（images/、labels/、data.yaml）"""
    for split in ("train", "val"):
        doc = json.loads((src / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for im in doc["images"]:
            (dst / "images" / split / im["file_name"]).write_bytes((src / "images" / split / im["file_name"]).read_bytes())
            w, h = im["width"], im["height"]
            lines = [
                f"{a['category_id'] - 1} {(a['bbox'][0] + a['bbox'][2] / 2) / w:.6f} {(a['bbox'][1] + a['bbox'][3] / 2) / h:.6f} "
                f"{a['bbox'][2] / w:.6f} {a['bbox'][3] / h:.6f}"
                for a in doc["annotations"] if a["image_id"] == im["id"]
            ]
            (dst / "labels" / split / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
    names = "".join(f"  {i}: {json.dumps(c, ensure_ascii=False)}\n" for i, c in enumerate(CLASSES))
    (dst / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\nnc: {len(CLASSES)}\nnames:\n{names}", encoding="utf-8")


@unittest.skipUnless(HAS_ULTRA, "没有安装 ultralytics")
class UltralyticsTrainTest(unittest.TestCase):
    def test_train_export_eval(self):
        from xlabel_worker import onnx_model, train, ul_train

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_dataset(tmp / "ds")
            coco_to_yolo(tmp / "ds", tmp / "yolo")
            base = {
                "engine": "ultralytics", "dataset": str(tmp / "ds"), "data_yolo": str(tmp / "yolo"), "model": "yolov8n.yaml",
                "classes": CLASSES, "task": "detect", "epochs": 1, "img_size": 96, "batch": 4, "device": "cpu", "workers": 0,
                "extra": "mosaic=0\namp=False",
            }
            out = tmp / "run"
            train.run({**base, "out": str(out)})
            self.assertTrue((out / "native" / "weights" / "best.pt").exists())
            self.assertTrue((out / "model.onnx").exists())
            meta = json.loads((out / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["task"], "detect")
            self.assertEqual(meta["classes"], CLASSES)
            self.assertEqual(meta["engine"], "ultralytics")
            metrics = [json.loads(ln) for ln in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(metrics), 1)
            self.assertIn("map50", metrics[0])
            ev = json.loads((out / "eval.json").read_text(encoding="utf-8"))
            self.assertIn("val", ev["splits"])
            self.assertNotIn("reevaluated", ev)

            # 训好的 ONNX 走通用解码
            det, info = onnx_model.load(out / "model.onnx")
            self.assertEqual(det.task, "detect")
            from PIL import Image

            res = det.predict_full(Image.fromarray(np.zeros((120, 160, 3), dtype=np.uint8)), 0.0)
            self.assertIsInstance(res, list)

            # 接着训（起点是上次的 best.pt）
            out2 = tmp / "run2"
            train.run({**base, "out": str(out2), "init_weights": str(out / "native" / "weights" / "best.pt")})
            self.assertTrue((out2 / "model.onnx").exists())

    def test_extra_params(self):
        from xlabel_worker.ul_train import _parse_extra

        self.assertEqual(
            _parse_extra("cos_lr=True\nmosaic = 0.5\npatience=30  # 早停\ndata=x.yaml\noptimizer='AdamW'\n"),
            {"cos_lr": True, "mosaic": 0.5, "patience": 30, "optimizer": "AdamW"},
        )


if __name__ == "__main__":
    unittest.main()
