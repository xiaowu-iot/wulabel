"""批量识别的视频截图：框、轮廓、类别和分数画上去，分类写在左上角，框贴着上边时标签放进框里。

运行：python -m unittest discover -s worker/tests
"""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from xlabel_worker.serve import draw


class DrawTest(unittest.TestCase):
    def test_draw_boxes_polys_tags(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "帧.jpg"
            Image.new("RGB", (640, 360), (40, 40, 40)).save(src)
            dst = Path(d) / "扇叶损坏_00m12s_0.93.jpg"
            items = [
                {"label": "扇叶损坏", "score": 0.93, "color": "#e5534b", "box": [100, 0, 200, 150]},
                {"label": "缺失配件", "score": 0.71, "color": None, "box": [350, 120, 120, 100],
                 "poly": [350, 120, 470, 120, 470, 220, 350, 220]},
                {"label": "全新商品", "score": 0.88, "color": "#2fb67c"},
            ]
            draw({"src": str(src), "dst": str(dst), "items": items})
            self.assertTrue(dst.exists())
            im = Image.open(dst).convert("RGB")
            self.assertEqual(im.size, (640, 360))
            # 框贴着上边：标签放在框里面，左上角是标签底色（红）
            r, g, b = im.getpixel((101, 1))
            self.assertGreater(r, 150)
            self.assertLess(g, 120)
            # 分类标签写在左上角（绿）
            r, g, b = im.getpixel((7, 7))
            self.assertGreater(g, 120)

    def test_draw_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.jpg"
            Image.new("RGB", (100, 80), (0, 0, 0)).save(src)
            dst = Path(d) / "b.jpg"
            draw({"src": str(src), "dst": str(dst), "items": []})
            self.assertTrue(dst.exists())


if __name__ == "__main__":
    unittest.main()
