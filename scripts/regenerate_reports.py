from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.benchmark import (  # noqa: E402
    _apply_top_k_thresholds,
    _primary_label_operating_metrics,
    _top1_outcomes_by_primary_label,
    save_summary,
)
from yolo_benchmark.common import DEFAULT_CONFIG, load_config, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild benchmark reports from existing predictions.csv files"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--backend", choices=("pytorch", "onnx", "both"), default="both"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    classes = list(config["classes"])
    top_k = int(config["top_k"])
    threshold_values = np.asarray(
        [float(config["verification_thresholds"][name]) for name in classes]
    )
    backends = ["pytorch", "onnx"] if args.backend == "both" else [args.backend]

    for backend in backends:
        backend_dir = config["output_dir"] / backend
        summary_path = backend_dir / "summary.json"
        if not summary_path.exists():
            print(f"Skipped {backend}: no existing summary at {summary_path.resolve()}")
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)

        for result in results:
            run_name = Path(result["model"]).stem
            model_dir = backend_dir / run_name
            with (model_dir / "predictions.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            targets = [classes.index(row["true_label"]) for row in rows]
            scores = np.asarray(
                [
                    [float(row[f"score_{name}"]) for name in classes]
                    for row in rows
                ],
                dtype=float,
            )
            accepted = _apply_top_k_thresholds(scores, threshold_values, top_k)
            predictions = np.argmax(scores, axis=1)
            open_predictions = np.asarray(
                [
                    prediction
                    if accepted[row_index, prediction]
                    else len(classes)
                    for row_index, prediction in enumerate(predictions)
                ]
            )
            for row_index, (row, target, prediction) in enumerate(
                zip(rows, targets, predictions)
            ):
                row["predicted_label"] = (
                    classes[prediction]
                    if accepted[row_index, prediction]
                    else "unknown"
                )
                row["accepted_labels"] = ";".join(
                    name
                    for index, name in enumerate(classes)
                    if accepted[row_index, index]
                ) or "unknown"
                row["expected_class_accepted"] = str(
                    bool(accepted[row_index, target])
                )
                for index, name in enumerate(classes):
                    row[f"accepted_{name}"] = str(bool(accepted[row_index, index]))
            with (model_dir / "predictions.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result.update(
                _primary_label_operating_metrics(targets, accepted, classes)
            )
            result["top_k"] = top_k
            result["top1_outcomes_by_primary_label"] = (
                _top1_outcomes_by_primary_label(
                    targets, predictions, open_predictions, classes
                )
            )
            write_json(model_dir / "metrics.json", result)

        save_summary(results, backend_dir)
        print(f"Rebuilt reports: {backend_dir.resolve()}")


if __name__ == "__main__":
    main()
