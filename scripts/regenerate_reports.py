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
    _primary_label_operating_metrics,
    save_summary,
)
from yolo_benchmark.common import DEFAULT_CONFIG, load_config, write_json  # noqa: E402


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


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
            accepted = np.asarray(
                [
                    [_as_bool(row[f"accepted_{name}"]) for name in classes]
                    for row in rows
                ],
                dtype=bool,
            )
            result.update(
                _primary_label_operating_metrics(targets, accepted, classes)
            )
            write_json(model_dir / "metrics.json", result)

        save_summary(results, backend_dir)
        print(f"Rebuilt reports: {backend_dir.resolve()}")


if __name__ == "__main__":
    main()
