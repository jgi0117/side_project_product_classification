from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.rtmdet_x import decide_top1


class RTMDetXDecisionTests(unittest.TestCase):
    def test_tv_and_laptop_use_max_score_and_book_can_win(self):
        result = decide_top1([(62, .4), (63, .5), (73, .6)])
        self.assertEqual(result["group_scores"]["computer"], .5)
        self.assertEqual(result["prediction"], "book")
        self.assertEqual(result["top_k"], 1)

    def test_other_and_no_passing_detection(self):
        self.assertEqual(decide_top1([(1, .9), (73, .1)])["prediction"], "other")
        self.assertEqual(decide_top1([(73, .05)])["prediction"], "book")
        result = decide_top1([(73, .049)])
        self.assertEqual(result["prediction"], "other")
        self.assertEqual(result["decision_reason"], "no_passing_detection")
        self.assertEqual(decide_top1([])["confidence"], 0)


if __name__ == "__main__":
    unittest.main()
