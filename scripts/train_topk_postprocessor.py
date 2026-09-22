"""Train and report an image-level classifier over frozen RTMDet detections."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.common import write_json  # noqa: E402
from yolo_benchmark.postprocessor import (TARGET_CLASSES, binary_metrics,
    extract_features, ranked_candidates, source_summary, topk_summary, truth_label)  # noqa: E402
from yolo_benchmark.reports import decide, percent, write_csv  # noqa: E402


def estimator(seed: int, inner_splits: int) -> GridSearchCV:
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=5000, random_state=seed)),
    ])
    return GridSearchCV(
        pipeline, {"classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(inner_splits, shuffle=True, random_state=seed + 1),
        n_jobs=1,
    )


def reorder_probabilities(probabilities, estimator_classes, target_classes):
    indices = [list(estimator_classes).index(label) for label in target_classes]
    return probabilities[:, indices]


def evaluate(truth, probabilities, classes):
    top1 = ranked_candidates(probabilities, classes, 1)
    top2 = ranked_candidates(probabilities, classes, 2)
    prediction = [row[0] for row in top1]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=classes, zero_division=0)
    per_class = [{"class": label, "support": int(support[index]),
                  "precision": float(precision[index]), "recall": float(recall[index]),
                  "f1": float(f1[index])} for index, label in enumerate(classes)]
    return {
        "top1_accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "top1": topk_summary(truth, top1),
        "top2": topk_summary(truth, top2),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=classes).tolist(),
        "per_class": per_class,
        "binary": [{"top_k": k, **binary_metrics(truth, candidates, label)}
                   for k, candidates in ((1, top1), (2, top2))
                   for label in TARGET_CLASSES[:2]],
    }, top1, top2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.model_output.exists():
        parser.error(f"model output already exists: {args.model_output}")

    snapshot = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    raw_path = args.run / "rtmdet-tiny/detections.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    expected = [(row["sha256"], row["source_label"]) for row in snapshot["images"]]
    actual = [(row["sha256"], row["source_label"]) for row in raw["images"]]
    if expected != actual or len(actual) != len(set(sha for sha, _ in actual)):
        raise ValueError("Detection records do not match the unique run manifest")

    x, feature_names = extract_features(raw["images"])
    truth = [truth_label(row["source_label"]) for row in raw["images"]]
    source_labels = [row["source_label"] for row in raw["images"]]
    classes = list(TARGET_CLASSES)
    y = np.asarray(truth)
    outer = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(y), len(classes)), dtype=np.float64)
    folds = np.zeros(len(y), dtype=np.int64)
    fold_settings = []
    for fold, (train_indices, test_indices) in enumerate(outer.split(x, y), start=1):
        search = estimator(args.seed + fold * 10, 4)
        search.fit(x[train_indices], y[train_indices])
        oof[test_indices] = reorder_probabilities(
            search.predict_proba(x[test_indices]), search.best_estimator_.classes_, classes)
        folds[test_indices] = fold
        fold_settings.append({"fold": fold, "train_images": len(train_indices),
                              "test_images": len(test_indices),
                              "best_c": search.best_params_["classifier__C"],
                              "inner_balanced_accuracy": float(search.best_score_)})

    trained, trained_top1, trained_top2 = evaluate(truth, oof, classes)
    baseline_top2 = []
    baseline_probabilities = np.zeros_like(oof)
    for index, image in enumerate(raw["images"]):
        decision = decide(image["detections"], {**snapshot["config"], "top_k": 2})
        baseline_top2.append(decision["candidates"])
        for rank, label in enumerate(decision["candidates"]):
            baseline_probabilities[index, classes.index(label)] = 2 - rank
    baseline, baseline_top1, _ = evaluate(truth, baseline_probabilities, classes)
    # Preserve threshold behavior: a baseline may emit only one candidate.
    baseline["top2"] = topk_summary(truth, baseline_top2)
    baseline["binary"] = [{"top_k": k, **binary_metrics(truth, candidates, label)}
                          for k, candidates in ((1, baseline_top1), (2, baseline_top2))
                          for label in TARGET_CLASSES[:2]]

    final_search = estimator(args.seed + 1000, 5)
    final_search.fit(x, y)
    bundle = {
        "schema_version": 1,
        "scope": "image-level postprocessor over frozen RTMDet COCO detections",
        "classes": classes,
        "feature_names": feature_names,
        "model": final_search.best_estimator_,
        "training": {"images": len(y), "class_counts": dict(Counter(truth)),
                     "seed": args.seed, "selected_c": final_search.best_params_["classifier__C"],
                     "cv_balanced_accuracy": float(final_search.best_score_),
                     "detections_sha256": sha256(raw_path),
                     "rtmdet_weights_sha256": raw["runtime"].get("weights_sha256"),
                     "effective_score_floor": raw["runtime"].get("effective_score_floor"),
                     "packages": {name: importlib.metadata.version(name)
                                  for name in ("numpy", "scikit-learn", "joblib")}},
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)

    args.output.mkdir(parents=True)
    predictions = []
    for index, image in enumerate(raw["images"]):
        predictions.append({
            "sha256": image["sha256"], "source_label": image["source_label"],
            "truth": truth[index], "fold": int(folds[index]),
            "top1": trained_top1[index][0], "top2": ",".join(trained_top2[index]),
            **{f"probability_{label}": float(oof[index, classes.index(label)]) for label in classes},
        })
    write_csv(args.output / "oof-predictions.csv", predictions)
    write_csv(args.output / "per-class.csv", trained["per_class"])
    write_csv(args.output / "binary-topk.csv", trained["binary"])
    source_rows = source_summary(source_labels, truth, trained_top1, trained_top2)
    write_csv(args.output / "by-source.csv", source_rows)

    metrics = {
        "scope": bundle["scope"], "fine_tunes_rtmdet": False,
        "annotation": "single folder label per image; no bounding boxes",
        "evaluation": "nested stratified 5-fold out-of-fold predictions",
        "external_test_set": False, "classes": classes,
        "images": len(y), "class_counts": dict(Counter(truth)),
        "source_counts": dict(Counter(source_labels)), "features": len(feature_names),
        "outer_folds": fold_settings, "final_selected_c": final_search.best_params_["classifier__C"],
        "baseline": baseline, "trained_postprocessor": trained,
        "by_source": source_rows, "detections_sha256": sha256(raw_path),
        "model_sha256": sha256(args.model_output),
        "model_path": str(args.model_output.as_posix()),
    }
    write_json(args.output / "metrics.json", metrics)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["all", *classes]
    x_positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series = [
        ("baseline top-1", baseline["top1"], -0.27, "#94a3b8"),
        ("trained top-1 (OOF)", trained["top1"], -0.09, "#2563eb"),
        ("baseline top-2", baseline["top2"], 0.09, "#5eead4"),
        ("trained top-2 (OOF)", trained["top2"], 0.27, "#0f766e"),
    ]
    for name, rows, offset, color in series:
        values = [100 * next(row["rate"] for row in rows if row["class"] == label) for label in labels]
        ax.bar(x_positions + offset, values, 0.18, label=name, color=color)
    ax.set(xticks=x_positions, xticklabels=labels, ylim=(0, 108),
           ylabel="Ground-truth candidate hit rate (%)",
           title="RTMDet frozen detections: baseline vs trained postprocessor")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(args.output / "topk-comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    matrix = np.asarray(trained["confusion_matrix"])
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(len(classes)):
        for column in range(len(classes)):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set(xticks=range(len(classes)), xticklabels=classes,
           yticks=range(len(classes)), yticklabels=classes,
           xlabel="Predicted", ylabel="Ground truth",
           title="Trained postprocessor: out-of-fold confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output / "confusion-matrix.png", dpi=160)
    plt.close(fig)

    baseline1 = next(row["rate"] for row in baseline["top1"] if row["class"] == "all")
    baseline2 = next(row["rate"] for row in baseline["top2"] if row["class"] == "all")
    trained1 = next(row["rate"] for row in trained["top1"] if row["class"] == "all")
    trained2 = next(row["rate"] for row in trained["top2"] if row["class"] == "all")
    lines = [
        "# RTMDet 탐지 특징 학습 및 top-k 보고서", "",
        "RTMDet-tiny의 COCO 가중치는 고정하고, 저장된 탐지 결과의 클래스별 최대 점수와 탐지 수를 특징으로 사용해",
        "computer/book/other 이미지 분류기를 학습했습니다. 탐지기 fine-tuning 결과가 아닙니다.", "",
        "## 핵심 결과", "",
        "| 모델 | Top-1 | Top-2 정답 포함률 |",
        "|---|---:|---:|",
        f"| 기존 규칙 | {percent(baseline1)} | {percent(baseline2)} |",
        f"| 학습 후처리기 (OOF) | {percent(trained1)} | {percent(trained2)} |", "",
        f"Top-1은 {(trained1 - baseline1) * 100:.2f}%p, top-2는 {(trained2 - baseline2) * 100:.2f}%p 상승했습니다.",
        "학습 지표는 각 이미지를 학습에서 제외한 5-fold out-of-fold 예측으로 계산했습니다.", "",
        "![Top-k 비교](topk-comparison.png)", "", "## 클래스별 out-of-fold 결과", "",
        "| 클래스 | 이미지 | Precision | Recall | F1 | Top-2 포함률 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in trained["per_class"]:
        top2_rate = next(item["rate"] for item in trained["top2"] if item["class"] == row["class"])
        lines.append(f"| {row['class']} | {row['support']} | {percent(row['precision'])} | "
                     f"{percent(row['recall'])} | {percent(row['f1'])} | {percent(top2_rate)} |")
    lines += ["", "![혼동행렬](confusion-matrix.png)", "", "## 등록 카테고리 수락 지표", "",
              "| 방식 | 카테고리 | k | Precision | Recall | FPR | FNR |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, result in (("기존 규칙", baseline), ("학습 후처리기 (OOF)", trained)):
        for row in result["binary"]:
            lines.append(f"| {name} | {row['class']} | {row['top_k']} | "
                         f"{percent(row['precision'])} | {percent(row['recall'])} | "
                         f"{percent(row['fpr'])} | {percent(row['fnr'])} |")
    lines += ["", "학습 후처리기의 top-1은 recall을 높였지만 FPR도 증가했습니다. "
              "Top-2는 항상 두 후보를 내므로 FPR이 크게 증가하며 검토 후보 제시에만 사용해야 합니다.",
              "", "## 해석 범위", "",
              "- 원본 451개 파일에서 정확한 중복을 제거한 372장을 사용했습니다.",
              "- 폴더의 단일 라벨만 사용했습니다. 객체별 박스 주석은 없습니다.",
              "- 별도 외부 테스트셋이 없으므로 새로운 촬영 환경의 성능을 보장하지 않습니다.",
              "- 최종 모델은 372장 전체로 다시 학습했으며, 보고된 성능은 해당 최종 모델의 학습 정확도가 아니라 OOF 추정치입니다.",
              "- Top-2는 후보 두 개 중 정답 포함 여부이며 단일 승인 정확도가 아닙니다.", "",
              "## 산출물", "",
              "- `metrics.json`: 설정, fold별 선택값, 전체 지표, 해시",
              "- `oof-predictions.csv`: 경로를 제외한 이미지 해시별 OOF 예측",
              "- `per-class.csv`, `binary-topk.csv`, `by-source.csv`: 세부 지표",
              f"- `{args.model_output.as_posix()}`: 전체 데이터로 학습한 배포용 후처리기", ""]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
