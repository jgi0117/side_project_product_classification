import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_photo_report import binary_metrics


class PhotoReportTests(unittest.TestCase):
    def test_candidate_false_accepts_use_all_other_classes(self):
        rows = [
            {"truth": "book", "candidates": ["other", "book"]},
            {"truth": "book", "candidates": ["other"]},
            {"truth": "computer", "candidates": ["computer", "book"]},
            {"truth": "other", "candidates": ["other", "book"]},
            {"truth": "other", "candidates": ["other"]},
        ]
        b = binary_metrics(rows, "book")
        self.assertEqual([b[k] for k in ["tp", "fp", "fn", "tn"]], [1, 2, 1, 1])
        self.assertEqual(b["accuracy"], 2 / 5)
        self.assertEqual(b["precision"], 1 / 3)
        self.assertEqual(b["recall"], 1 / 2)
        self.assertEqual(b["fpr"], 2 / 3)

    def test_undefined_precision_and_negative_support(self):
        b = binary_metrics([{"truth": "book", "candidates": ["other"]}], "book")
        self.assertIsNone(b["precision"])
        self.assertIsNone(b["fpr"])
        self.assertEqual(b["recall"], 0)


if __name__ == "__main__":
    unittest.main()
