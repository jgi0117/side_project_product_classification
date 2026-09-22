from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.postprocessor import (binary_metrics, extract_features,
                                           ranked_candidates, truth_label)


class PostprocessorTests(unittest.TestCase):
    def test_features_use_max_score_and_log_count(self):
        images = [{"detections": [
            {"label": "book", "score": .2}, {"label": "book", "score": .8},
            {"label": "laptop", "score": .4},
        ]}]
        features, names = extract_features(images)
        self.assertEqual(features.shape, (1, 160))
        self.assertEqual(features[0, names.index("max_score:book")], .8)
        self.assertAlmostEqual(features[0, names.index("log1p_count:book")], np.log(3))

    def test_rank_and_binary_metrics(self):
        probabilities = np.array([[.7, .2, .1], [.1, .3, .6]])
        classes = ["computer", "book", "other"]
        top2 = ranked_candidates(probabilities, classes, 2)
        self.assertEqual(top2, [["computer", "book"], ["other", "book"]])
        metrics = binary_metrics(["computer", "other"], top2, "book")
        self.assertEqual([metrics[key] for key in ("tp", "fp", "fn", "tn")], [0, 2, 0, 0])

    def test_unknown_source_is_other(self):
        self.assertEqual(truth_label("bed"), "other")
        self.assertEqual(truth_label("book"), "book")


if __name__ == "__main__":
    unittest.main()
