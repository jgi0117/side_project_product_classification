from __future__ import annotations

import sys
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image
from yolo_benchmark.common import load_config
from yolo_benchmark.data import discover_images
from yolo_benchmark.reports import decide, evaluate, render_model, render_summary


@contextmanager
def temporary_workspace():
    parent = (ROOT / "outputs").resolve()
    parent.mkdir(exist_ok=True)
    folder = parent / ("test-" + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent or folder.is_symlink():
            raise ValueError("Unsafe test cleanup path")
        shutil.rmtree(folder)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_tv_and_laptop_merge_without_summing_confidences(self):
        result = decide([{"label": "laptop", "score": .4}, {"label": "tv", "score": .5},
                         {"label": "book", "score": .7}], self.config)
        self.assertEqual(result["scores"]["computer"], .5)
        self.assertEqual(result["prediction"], "book")

    def test_other_competes_and_no_detection_rejects(self):
        result = decide([{"label": "bicycle", "score": .9}, {"label": "book", "score": .1}], self.config)
        self.assertEqual(result["prediction"], "other")
        self.assertEqual(result["raw_top1"], "bicycle")
        self.assertEqual(decide([], self.config)["prediction"], "other")
        self.assertEqual(decide([{"label": "book", "score": .049}], self.config)["prediction"], "other")
        self.assertEqual(decide([{"label": "book", "score": .05}], self.config)["prediction"], "book")

    def test_hand_calculated_matrix_and_error_denominators(self):
        truth = ["computer", "computer", "book", "book", "other", "other"]
        prediction = ["computer", "book", "book", "other", "computer", "other"]
        rows = [{"truth": t, "prediction": p, "source_label": t} for t, p in zip(truth, prediction)]
        metrics = evaluate(rows, self.config)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["image_false_accept_rate"], .5)
        self.assertEqual(metrics["image_false_reject_rate"], .25)
        for row in metrics["per_class"]:
            self.assertEqual(row["precision"], .5)
            self.assertEqual(row["recall"], .5)
            self.assertEqual(row["false_accept_rate"], .25)
        self.assertIsNone(metrics["session_false_accept_rate"])
        self.assertEqual(metrics["criteria"][2]["status"], "N/A")

    def test_missing_negative_support_is_not_zero_error(self):
        metrics = evaluate([{"truth": "book", "prediction": "book", "source_label": "book"}], self.config)
        self.assertIsNone(metrics["image_false_accept_rate"])
        self.assertIsNone(metrics["per_class"][0]["precision"])
        self.assertIsNone(metrics["macro"]["recall"])

    def test_data_includes_unknown_categories_and_keeps_primary_folder(self):
        # Explicit workspace temp path avoids restricted Windows global temp.
        with temporary_workspace() as folder:
            source = Path(folder)
            for index, label in enumerate(["laptop", "tv", "book", "bike", "guitar", "custom_negative"]):
                child = source / label / "book"  # nested name must not overwrite parent label
                child.mkdir(parents=True)
                Image.new("RGB", (8, 8), (index * 40, 0, 0)).save(child / "image.png")
            items, invalid = discover_images(source, self.config["classes"])
            self.assertEqual(len(items), 6)
            self.assertEqual(invalid, 0)
            self.assertEqual(sorted(i.label for i in items), ["bicycle", "book", "computer", "computer", "custom_negative", "guitar"])
            filtered, _ = discover_images(source, self.config["classes"], include_other=False)
            self.assertEqual(len(filtered), 3)

    def test_report_artifacts(self):
        with temporary_workspace() as folder:
            output = Path(folder)
            raw = {"model": "test-model", "runtime": {"latency_mean_ms": 2, "fps": 500, "model_size_mib": 1},
                   "images": [{"path": "book/image.png", "source_label": "book",
                               "detections": [{"label": "book", "score": .9, "xyxy": [0, 0, 8, 8]}]}]}
            metrics = render_model(raw, self.config, output / "test-model")
            render_summary([metrics], {}, output)
            for name in ["metrics.json", "predictions.csv", "per_class.csv", "confusion_matrix.csv",
                         "confusion_matrix.png", "criteria.csv", "by_source.csv", "raw_coco_outcomes.csv", "report.md"]:
                self.assertTrue((output / "test-model" / name).is_file(), name)
            self.assertTrue((output / "report.html").is_file())
            self.assertEqual(metrics["accuracy"], 1)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    unittest.main()
