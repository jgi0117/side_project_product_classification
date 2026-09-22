"""Compare top-1/2 on identical cached RTMDet detections; never runs a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.common import write_json
from yolo_benchmark.reports import decide, percent, ratio, write_csv


def binary_metrics(rows, label):
    tp = sum(r["truth"] == label and label in r["candidates"] for r in rows)
    fp = sum(r["truth"] != label and label in r["candidates"] for r in rows)
    fn = sum(r["truth"] == label and label not in r["candidates"] for r in rows)
    tn = len(rows) - tp - fp - fn
    return {"class": label, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": ratio(tp + tn, len(rows)), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "fpr": ratio(fp, fp + tn), "fnr": ratio(fn, tp + fn)}


def compare(raw, config):
    rows = []
    for item in raw["images"]:
        decision = decide(item["detections"], {**config, "top_k": 2})
        truth = item["source_label"] if item["source_label"] in config["classes"] else "other"
        rows.append({"source_label": item["source_label"], "truth": truth,
                     "candidates": decision["candidates"],
                     "top1_hit": truth == decision["prediction"],
                     "top2_hit": truth in decision["candidates"]})
    if not rows:
        raise ValueError("No images to compare")
    summaries = []
    for label in ["all", *config["classes"], "other"]:
        subset = rows if label == "all" else [r for r in rows if r["truth"] == label]
        hits1 = sum(r["top1_hit"] for r in subset)
        hits2 = sum(r["top2_hit"] for r in subset)
        summaries.append({"class": label, "images": len(subset),
                          "top1": ratio(hits1, len(subset)), "top2": ratio(hits2, len(subset)),
                          "recovered": hits2 - hits1,
                          "gain_pp": ratio(100 * (hits2 - hits1), len(subset))})
    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run folder containing run.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/rtmdet-topk/results")
    args = parser.parse_args()
    snapshot = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    raw_path = args.run / "rtmdet-tiny/detections.json"
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    if raw["model"] != "rtmdet-tiny" or "rtmdet-tiny" not in snapshot["selected_models"]:
        raise ValueError("Expected a completed rtmdet-tiny run")
    expected = [(r["path"], r["source_label"], r["sha256"]) for r in snapshot["images"]]
    actual = [(r["path"], r["source_label"], r["sha256"]) for r in raw["images"]]
    if expected != actual:
        raise ValueError("Detections do not match the run image manifest")
    rows, summaries = compare(raw, snapshot["config"])
    binary = []
    for k in (1, 2):
        selected = [{**r, "candidates": r["candidates"][:k]} for r in rows]
        for label in snapshot["config"]["classes"]:
            binary.append({"top_k": k, **binary_metrics(selected, label)})
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "comparison.csv", summaries)
    write_csv(args.output / "category-verification.csv", binary)
    # Publish aggregate results only: no original image paths in Git artifacts.
    write_json(args.output / "comparison.json", {
        "model": raw["model"], "smoke_test": snapshot["smoke_test"],
        "detections_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "settings": {key: snapshot["config"][key] for key in (
            "coco_class_mapping", "verification_thresholds", "other_threshold",
            "detection_confidence", "nms_iou", "device")},
        "runtime": raw["runtime"], "comparison": summaries,
        "source_counts": snapshot["evaluated_source_counts"],
        "category_verification": binary,
        "candidate_counts": {str(n): sum(len(r["candidates"]) == n for r in rows) for n in (1, 2)}})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(summaries))
    for key, offset, color in [("top1", -.18, "#2563eb"), ("top2", .18, "#0d9488")]:
        values = [100 * r[key] if r[key] is not None else 0 for r in summaries]
        bars = ax.bar(positions + offset, values, .36, label=key, color=color)
        ax.bar_label(bars, labels=[percent(r[key]) for r in summaries], padding=3)
    ax.set(xticks=positions, xticklabels=[r["class"] for r in summaries], ylim=(0, 112),
           ylabel="Ground-truth candidate hit rate (%)",
           title="RTMDet-tiny: top-1 vs top-2" + (" (SMOKE ONLY)" if snapshot["smoke_test"] else ""))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "topk-comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# RTMDet-tiny top-k 비교 결과", "",
             "동일한 저장 탐지 결과에서 임계값을 통과한 MVP 그룹의 정답 포함률을 비교합니다.", "",
             "**SMOKE_ONLY: 샘플 결과이며 전체 성능 평가가 아닙니다.**" if snapshot["smoke_test"] else "전체 실행 결과 (원본 run.json의 smoke_test=false).",
             "", "| 정답 그룹 | 이미지 수 | Top-1 | Top-2 | 증가 (%p) | 추가 정답 수 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        gain = "N/A" if row["gain_pp"] is None else f"{row['gain_pp']:.2f}"
        lines.append(f"| {row['class']} | {row['images']} | {percent(row['top1'])} | {percent(row['top2'])} | {gain} | {row['recovered']} |")
    lines += ["", "![Top-k 비교](topk-comparison.png)", "",
              "Top-2는 후보를 최대 두 개 허용한 포함률입니다. 단일 예측 정확도, precision/F1, 운영 수락·거절 오류율의 개선을 뜻하지 않습니다.",
              "세 그룹 중 두 후보를 허용하므로 포함률은 구조적으로 상승할 수 있습니다. 학습이나 가중치 변경은 없습니다.", "",
              "[집계 CSV](comparison.csv) · [설정 및 원본 탐지 SHA-256](comparison.json)", ""]
    lines += ["", "## 등록 카테고리별 수락/거절", "",
              "| 카테고리 | k | Accuracy | Precision | Recall | FPR | FNR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in binary:
        lines.append(f"| {row['class']} | {row['top_k']} | " + " | ".join(
            percent(row[key]) for key in ("accuracy", "precision", "recall", "fpr", "fnr")) + " |")
    lines += ["", "폴더의 단일 정답 기준입니다. 사진에 여러 대상이 함께 있으면 오수락이 과대 집계될 수 있습니다.",
              "[카테고리별 CSV](category-verification.csv)", ""]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
