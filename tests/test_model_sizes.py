from __future__ import annotations

import sys
import unittest

from test_reports import temporary_workspace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from compare_model_sizes import compare
from run_inference import validate_config
from yolo_benchmark.common import load_config, write_json


class ModelSizeTests(unittest.TestCase):
    def test_config_has_matching_variants(self):
        config = load_config()
        validate_config(config)
        self.assertEqual([m["name"] for m in config["models"]],
                         ["rtmdet-tiny", "rtmdet-s", "rtmdet-m", "rtmdet-l", "rtmdet-x"])
        config["models"][1]["config"] = config["models"][0]["config"]
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_config(config)

    def test_comparison_requires_same_manifest_and_reports_top2_separately(self):
        with temporary_workspace() as run:
            config = load_config()
            records = [{"path": "book/a.jpg", "source_label": "book", "sha256": "a"},
                       {"path": "other/b.jpg", "source_label": "other", "sha256": "b"}]
            serial_config = {key: str(value) if isinstance(value, Path) else value
                             for key, value in config.items()}
            write_json(run / "run.json", {"config": serial_config, "images": records,
                       "selected_models": ["rtmdet-tiny", "rtmdet-s"],
                       "smoke_test": False, "evaluated_source_counts": {"book": 1, "other": 1}})
            runtime = {"latency_mean_ms": 10, "fps": 100, "model_size_mib": 50,
                       "weights_sha256": "test"}
            detections = [[{"label": "bicycle", "score": .9},
                           {"label": "book", "score": .8}], []]
            for name in ("rtmdet-tiny", "rtmdet-s"):
                write_json(run / name / "detections.json", {"model": name, "runtime": runtime,
                           "images": [{**item, "detections": det} for item, det in zip(records, detections)]})
            _, rows = compare(run)
            self.assertEqual([(row["model"], row["top_k"]) for row in rows],
                             [("rtmdet-tiny", 1), ("rtmdet-tiny", 2),
                              ("rtmdet-s", 1), ("rtmdet-s", 2)])
            self.assertEqual(rows[0]["candidate_hit_rate"], .5)
            self.assertEqual(rows[1]["candidate_hit_rate"], 1)
            self.assertEqual(rows[0]["book_recall"], 0)
            self.assertEqual(rows[1]["book_recall"], 1)
            self.assertEqual(rows[1]["mean_candidates"], 1.5)
            self.assertEqual(rows[0]["images"], 2)
            self.assertEqual(rows[0]["model_size_mib"], 50)
            altered = [{**records[0], "sha256": "wrong", "detections": detections[0]},
                       {**records[1], "detections": detections[1]}]
            write_json(run / "rtmdet-s" / "detections.json", {"model": "rtmdet-s",
                       "runtime": runtime, "images": altered})
            with self.assertRaisesRegex(ValueError, "manifest"):
                compare(run)


if __name__ == "__main__":
    unittest.main()
