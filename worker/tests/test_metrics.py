"""运行：python -m unittest discover -s worker/tests"""

import unittest

import numpy as np

from xlabel_worker.metrics import ImageGt, ImagePred, average_precision, box_iou, evaluate


def gt(boxes, labels, image_id=0):
    return ImageGt(np.array(boxes, float).reshape(-1, 4), np.array(labels, int), image_id)


def pred(boxes, scores, labels):
    return ImagePred(np.array(boxes, float).reshape(-1, 4), np.array(scores, float), np.array(labels, int))


class MetricsTest(unittest.TestCase):
    def test_iou(self):
        a = np.array([[0, 0, 10, 10]], float)
        b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]], float)
        np.testing.assert_allclose(box_iou(a, b), [[50 / 150, 0]])

    def test_perfect(self):
        g = [gt([[0, 0, 10, 10], [20, 20, 40, 40]], [0, 1])]
        p = [pred([[0, 0, 10, 10], [20, 20, 40, 40]], [0.9, 0.8], [0, 1])]
        r = evaluate(p, g, 2)
        self.assertAlmostEqual(r.map50, 1.0)
        self.assertAlmostEqual(r.map, 1.0)
        self.assertEqual(r.errors, [])
        self.assertEqual(r.confusion, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])

    def test_half_recall(self):
        # 2 个真实框只找到 1 个：AP = 召回率 0~0.5 处精确率 1，其余 0 → 51/101
        g = [gt([[0, 0, 10, 10], [50, 50, 60, 60]], [0, 0], image_id=7)]
        p = [pred([[0, 0, 10, 10]], [0.9], [0])]
        r = evaluate(p, g, 1)
        self.assertAlmostEqual(r.map50, 51 / 101, places=6)
        self.assertEqual(r.errors, [{"imageId": 7, "missed": 1, "extra": 0, "wrong": 0}])
        self.assertAlmostEqual(r.recall, 0.5)

    def test_wrong_class_and_false_positive(self):
        g = [gt([[0, 0, 10, 10]], [0])]
        p = [pred([[0, 0, 10, 10], [100, 100, 120, 120]], [0.9, 0.8], [1, 0])]
        r = evaluate(p, g, 2)
        self.assertEqual(r.confusion[0][1], 1)  # 真实 0 被认成 1
        self.assertEqual(r.confusion[2][0], 1)  # 背景被认成 0
        self.assertEqual(r.errors[0]["wrong"], 1)
        self.assertEqual(r.errors[0]["extra"], 1)
        self.assertEqual(r.per_class[1]["ap50"], None)  # 类别 1 没有真实框

    def test_low_score_ignored_in_confusion(self):
        g = [gt([[0, 0, 10, 10]], [0])]
        p = [pred([[0, 0, 10, 10]], [0.1], [0])]
        r = evaluate(p, g, 1, conf=0.25)
        self.assertAlmostEqual(r.map50, 1.0)  # mAP 不受置信度阈值影响
        self.assertEqual(r.errors[0]["missed"], 1)

    def test_ap_ordering(self):
        # 高分的是错的：精确率先 0 后 0.5
        ap = average_precision(np.array([False, True]), np.array([0.9, 0.5]), 1)
        self.assertAlmostEqual(ap, 0.5)


if __name__ == "__main__":
    unittest.main()
