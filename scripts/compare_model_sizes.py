"""Compare completed RTMDet variants on the identical saved image manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.common import write_json
from yolo_benchmark.reports import decide, percent, ratio, write_csv


def candidate_metrics(rows, label, top_k):
    tp = sum(row["truth"] == label and label in row["candidates"][:top_k] for row in rows)
    fp = sum(row["truth"] != label and label in row["candidates"][:top_k] for row in rows)
    support = sum(row["truth"] == label for row in rows)
    return {"recall": ratio(tp, support), "precision": ratio(tp, tp + fp),
            "fpr": ratio(fp, len(rows) - support)}


def compare(run: Path):
    snapshot = json.loads((run / "run.json").read_text(encoding="utf-8"))
    expected = [(item["path"], item["source_label"], item["sha256"])
                for item in snapshot["images"]]
    names = snapshot["selected_models"]
    if not expected or not names or len(names) != len(set(names)):
        raise ValueError("Run must contain images and unique selected models")
    config = snapshot["config"]
    specs = {spec["name"]: spec for spec in config["models"]}
    results = []
    for name in names:
        if name not in specs or specs[name]["adapter"] != "rtmdet":
            raise ValueError(f"Unknown RTMDet model in run: {name}")
        raw_path = run / name / "detections.json"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Incomplete model {name}: {raw_path}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        actual = [(item["path"], item["source_label"], item["sha256"])
                  for item in raw["images"]]
        if raw["model"] != name or actual != expected:
            raise ValueError(f"{name}: detections do not match run manifest")
        rows = []
        for item in raw["images"]:
            decision = decide(item["detections"], {**config, "top_k": 2})
            rows.append({"truth": item["source_label"] if item["source_label"] in config["classes"] else "other",
                         "source_label": item["source_label"], **decision})
        runtime = raw["runtime"]
        for top_k in (1, 2):
            classes = {label: candidate_metrics(rows, label, top_k)
                       for label in (*config["classes"], "other")}
            results.append({"model": name, "config": specs[name]["config"], "top_k": top_k,
                            "images": len(rows),
                            "candidate_hit_rate": ratio(sum(row["truth"] in row["candidates"][:top_k]
                                                            for row in rows), len(rows)),
                            "mean_candidates": sum(min(len(row["candidates"]), top_k)
                                                   for row in rows) / len(rows),
                            **{f"{label}_{metric}": value for label, measures in classes.items()
                               for metric, value in measures.items()},
                            "latency_mean_ms": runtime["latency_mean_ms"],
                            "fps": runtime["fps"], "model_size_mib": runtime["model_size_mib"],
                            "weights_sha256": runtime["weights_sha256"]})
    return snapshot, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed run folder containing run.json")
    args = parser.parse_args()
    if not (args.run / "run.json").is_file():
        parser.error(f"No completed run at {args.run}; run_inference.py must finish successfully first")
    try:
        snapshot, results = compare(args.run)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    output = args.run / "model-size-comparison"
    output.mkdir(exist_ok=True)
    write_csv(output / "comparison.csv", results)
    write_json(output / "comparison.json", {
        "smoke_test": snapshot["smoke_test"],
        "evaluated_source_counts": snapshot["evaluated_source_counts"],
        "settings": {key: snapshot["config"][key] for key in (
            "coco_class_mapping", "verification_thresholds", "other_threshold",
            "detection_confidence", "nms_iou", "device", "speed_warmup", "speed_repeats")},
        "comparison": results})
    lines = ["# RTMDet model size comparison", "",
             "SMOKE_ONLY: subset results; do not interpret as full dataset performance." if snapshot["smoke_test"]
             else "Full run over the saved image manifest.", "",
             "Top-k hit rate is the fraction of images whose folder label is among up to k candidates.",
             "For k=1 this equals single-label accuracy. For k=2 it is candidate inclusion, not classification accuracy.",
             "Precision and FPR treat membership in the candidate list as a positive decision for each class.",
             "Latency is mean milliseconds per decoded image, batch size 1, on the saved device.", "",
             "| Model | k | Images | Hit rate | Mean candidates | Computer recall | Book recall | Other recall | Computer precision | Book precision | Computer FPR | Book FPR | ms/image | FPS | MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append("| " + " | ".join([
            row["model"], str(row["top_k"]), str(row["images"]),
            percent(row["candidate_hit_rate"]), f"{row['mean_candidates']:.2f}",
            *[percent(row[key]) for key in ("computer_recall", "book_recall", "other_recall",
                                            "computer_precision", "book_precision",
                                            "computer_fpr", "book_fpr")],
            f"{row['latency_mean_ms']:.2f}", f"{row['fps']:.2f}", f"{row['model_size_mib']:.2f}"]) + " |")
    lines += ["", "These are folder-label image metrics, not COCO box mAP or session error rates.",
              "Checkpoint hashes and exact settings are in [comparison.json](comparison.json).", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "README.md")


if __name__ == "__main__":
    main()
