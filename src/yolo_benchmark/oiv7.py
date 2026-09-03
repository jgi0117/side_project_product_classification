from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .common import write_json
from .data import ImageItem


def _save_class_summary(
    rows: list[dict[str, Any]], classes: list[str], output_dir: Path
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for label in classes:
        class_rows = [row for row in rows if row["true_label"] == label]
        passed = sum(bool(row["expected_label_pass"]) for row in class_rows)
        top1_counts = Counter(row["model_top1_label"] for row in class_rows)
        common_label, common_count = top1_counts.most_common(1)[0]
        summary_rows.append(
            {
                "class": label,
                "images": len(class_rows),
                "pass_count": passed,
                "reject_count": len(class_rows) - passed,
                "pass_rate": passed / len(class_rows),
                "mean_top1_score": float(
                    np.mean([row["model_top1_score"] for row in class_rows])
                ),
                "most_common_top1_label": common_label,
                "most_common_top1_count": common_count,
            }
        )

    with (output_dir / "class_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def _plot_verification_by_class(
    summary_rows: list[dict[str, Any]], output_dir: Path
) -> None:
    labels = [row["class"] for row in summary_rows]
    passed = np.array([row["pass_count"] for row in summary_rows])
    rejected = np.array([row["reject_count"] for row in summary_rows])
    totals = passed + rejected

    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(labels))
    axis.bar(positions, passed, label="Pass", color="#59A14F")
    axis.bar(positions, rejected, bottom=passed, label="Reject", color="#E15759")
    for index, (pass_count, total) in enumerate(zip(passed, totals)):
        axis.text(
            index,
            total + max(totals) * 0.02,
            f"{pass_count}/{total} ({pass_count / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Images")
    axis.set_title("Zero-shot Verification by Folder Label")
    axis.set_ylim(0, max(totals) * 1.16)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "verification_by_class.png", dpi=220)
    plt.close(figure)


def _top1_matrix(
    rows: list[dict[str, Any]],
    classes: list[str],
    target_model_labels: dict[str, str],
    limit: int = 12,
) -> tuple[list[str], np.ndarray]:
    overall = Counter(row["model_top1_label"] for row in rows)
    selected = [label for label, _ in overall.most_common(limit)]
    for label in target_model_labels.values():
        if label in overall and label not in selected:
            selected.append(label)
    has_other = len(selected) < len(overall)
    columns = selected + (["Other"] if has_other else [])
    matrix = np.zeros((len(classes), len(columns)), dtype=int)
    for row in rows:
        row_index = classes.index(row["true_label"])
        label = row["model_top1_label"]
        if label in selected:
            column_index = selected.index(label)
        else:
            column_index = len(columns) - 1
        matrix[row_index, column_index] += 1
    return columns, matrix


def _plot_top1_by_true_class(
    rows: list[dict[str, Any]],
    classes: list[str],
    target_model_labels: dict[str, str],
    output_dir: Path,
) -> None:
    columns, matrix = _top1_matrix(rows, classes, target_model_labels)
    width = max(11, len(columns) * 0.85)
    figure, axis = plt.subplots(figsize=(width, 4.8))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if value:
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() * 0.55 else "black",
                    fontsize=9,
                )
    axis.set_xticks(range(len(columns)), columns, rotation=40, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Unfiltered model top-1 label")
    axis.set_ylabel("Folder label")
    axis.set_title("Top-1 Predictions Across the Full Pretrained Vocabulary")
    figure.colorbar(image, ax=axis, label="Images")
    figure.tight_layout()
    figure.savefig(output_dir / "top1_by_true_class.png", dpi=220)
    plt.close(figure)


def _plot_book_top1_labels(
    rows: list[dict[str, Any]], output_dir: Path, limit: int = 12
) -> None:
    book_rows = [row for row in rows if row["true_label"] == "book"]
    if not book_rows:
        return
    counts = Counter(row["model_top1_label"] for row in book_rows)
    shown = counts.most_common(limit)
    shown_total = sum(count for _, count in shown)
    if shown_total < len(book_rows):
        shown.append(("Other", len(book_rows) - shown_total))
    labels = [label for label, _ in shown][::-1]
    values = [count for _, count in shown][::-1]

    figure, axis = plt.subplots(figsize=(9, max(4.5, len(labels) * 0.42)))
    axis.barh(labels, values, color="#4E79A7")
    for index, value in enumerate(values):
        axis.text(value + 0.08, index, str(value), va="center")
    axis.set_xlabel("Book-folder images")
    axis.set_title("Book Images: Unfiltered Model Top-1 Labels")
    axis.set_xlim(0, max(values) * 1.2)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "book_top1_labels.png", dpi=220)
    plt.close(figure)


def _rank_matrix(
    rows: list[dict[str, Any]],
    classes: list[str],
    rank: int,
    limit: int = 12,
) -> tuple[list[str], np.ndarray]:
    label_field = f"model_top{rank}_label"
    overall = Counter(row[label_field] for row in rows)
    selected = [label for label, _ in overall.most_common(limit)]
    has_other = len(selected) < len(overall)
    columns = selected + (["Other"] if has_other else [])
    matrix = np.zeros((len(classes), len(columns)), dtype=int)
    for row in rows:
        row_index = classes.index(row["true_label"])
        label = row[label_field]
        column_index = selected.index(label) if label in selected else len(columns) - 1
        matrix[row_index, column_index] += 1
    return columns, matrix


def _plot_rank_by_true_class(
    rows: list[dict[str, Any]], classes: list[str], output_dir: Path, rank: int
) -> None:
    columns, matrix = _rank_matrix(rows, classes, rank)
    width = max(11, len(columns) * 0.85)
    figure, axis = plt.subplots(figsize=(width, 4.8))
    image = axis.imshow(matrix, cmap="Purples", aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if value:
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() * 0.55 else "black",
                    fontsize=9,
                )
    axis.set_xticks(range(len(columns)), columns, rotation=40, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel(f"Unfiltered model rank-{rank} label")
    axis.set_ylabel("Folder label")
    axis.set_title(f"Rank-{rank} Predictions Across the Full Pretrained Vocabulary")
    figure.colorbar(image, ax=axis, label="Images")
    figure.tight_layout()
    figure.savefig(output_dir / f"top{rank}_by_true_class.png", dpi=220)
    plt.close(figure)


def _plot_book_rank_labels(
    rows: list[dict[str, Any]], output_dir: Path, rank: int, limit: int = 12
) -> None:
    book_rows = [row for row in rows if row["true_label"] == "book"]
    if not book_rows:
        return
    label_field = f"model_top{rank}_label"
    counts = Counter(row[label_field] for row in book_rows)
    shown = counts.most_common(limit)
    shown_total = sum(count for _, count in shown)
    if shown_total < len(book_rows):
        shown.append(("Other", len(book_rows) - shown_total))
    labels = [label for label, _ in shown][::-1]
    values = [count for _, count in shown][::-1]
    figure, axis = plt.subplots(figsize=(9, max(4.5, len(labels) * 0.42)))
    axis.barh(labels, values, color="#B07AA1")
    for index, value in enumerate(values):
        axis.text(value + 0.08, index, str(value), va="center")
    axis.set_xlabel("Book-folder images")
    axis.set_title(f"Book Images: Unfiltered Model Rank-{rank} Labels")
    axis.set_xlim(0, max(values) * 1.2)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / f"book_top{rank}_labels.png", dpi=220)
    plt.close(figure)


def _save_topk_label_summary(
    rows: list[dict[str, Any]], classes: list[str], output_dir: Path, top_k: int
) -> None:
    summary = []
    for true_label in classes:
        class_rows = [row for row in rows if row["true_label"] == true_label]
        for rank in range(1, top_k + 1):
            counts = Counter(row[f"model_top{rank}_label"] for row in class_rows)
            for predicted_label, count in counts.most_common():
                summary.append(
                    {
                        "true_label": true_label,
                        "rank": rank,
                        "predicted_label": predicted_label,
                        "count": count,
                        "rate": count / len(class_rows),
                    }
                )
    with (output_dir / "topk_label_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def _plot_confidence_by_class(
    rows: list[dict[str, Any]], classes: list[str], output_dir: Path
) -> None:
    scores = [
        [row["model_top1_score"] for row in rows if row["true_label"] == label]
        for label in classes
    ]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.boxplot(scores, tick_labels=classes, showmeans=True)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Unfiltered top-1 confidence")
    axis.set_title("Top-1 Detection Confidence by Folder Label")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "confidence_by_class.png", dpi=220)
    plt.close(figure)


def save_detector_visualizations(
    rows: list[dict[str, Any]],
    classes: list[str],
    target_model_labels: dict[str, str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    summary_rows = _save_class_summary(rows, classes, output_dir)
    _plot_verification_by_class(summary_rows, output_dir)
    _plot_top1_by_true_class(rows, classes, target_model_labels, output_dir)
    _plot_book_top1_labels(rows, output_dir)
    _plot_confidence_by_class(rows, classes, output_dir)
    top_k = 1
    while f"model_top{top_k + 1}_label" in rows[0]:
        top_k += 1
    _save_topk_label_summary(rows, classes, output_dir, top_k)
    for rank in range(2, top_k + 1):
        _plot_rank_by_true_class(rows, classes, output_dir, rank)
        _plot_book_rank_labels(rows, output_dir, rank)
    return summary_rows


def resolve_target_class_ids(
    model_names: dict[int, str] | list[str],
    target_model_labels: dict[str, str],
) -> dict[str, int]:
    """Resolve configured target labels against the checkpoint's class names."""
    if isinstance(model_names, list):
        indexed_names = dict(enumerate(model_names))
    else:
        indexed_names = {int(index): name for index, name in model_names.items()}

    normalized = {name.strip().casefold(): index for index, name in indexed_names.items()}
    resolved: dict[str, int] = {}
    for target, model_label in target_model_labels.items():
        index = normalized.get(model_label.strip().casefold())
        if index is None:
            raise ValueError(
                f"체크포인트에서 설정 클래스 {model_label!r}를 찾지 못했습니다."
            )
        resolved[target] = index
    return resolved


def _prediction_row(
    item: ImageItem,
    source: Path,
    result: Any,
    target_ids: dict[str, int],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    all_detections: list[tuple[int, float]] = []

    boxes = result.boxes
    if boxes is not None and len(boxes):
        detected_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy().astype(float)
        all_detections = list(zip(detected_ids.tolist(), confidences.tolist()))

    if all_detections:
        raw_class_id, raw_score = max(all_detections, key=lambda pair: pair[1])
        raw_label = str(result.names[raw_class_id])
    else:
        raw_class_id, raw_label, raw_score = -1, "unknown", 0.0

    target_by_id = {class_id: label for label, class_id in target_ids.items()}
    matched_target = target_by_id.get(raw_class_id)
    expected_label_pass = (
        matched_target == item.label
        and raw_score >= float(thresholds[item.label])
    )

    row: dict[str, Any] = {
        "file": item.source.relative_to(source).as_posix(),
        "true_label": item.label,
        "model_top1_class_id": raw_class_id,
        "model_top1_label": raw_label,
        "model_top1_score": raw_score,
        "decision": "pass" if expected_label_pass else "reject",
        "expected_label_pass": expected_label_pass,
        "all_detection_count": len(all_detections),
        "preprocess_ms": float(result.speed.get("preprocess", 0.0)),
        "inference_ms": float(result.speed.get("inference", 0.0)),
        "postprocess_ms": float(result.speed.get("postprocess", 0.0)),
    }
    row["total_pipeline_ms"] = (
        row["preprocess_ms"] + row["inference_ms"] + row["postprocess_ms"]
    )
    return row


def _model_size_mib(model: Any) -> tuple[str | None, float | None]:
    candidate = getattr(model, "ckpt_path", None)
    if not candidate:
        return None, None
    path = Path(str(candidate)).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.is_file():
        return str(candidate), None
    return str(path), path.stat().st_size / (1024**2)


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    def summarize(field: str, prefix: str) -> dict[str, float]:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        return {
            f"{prefix}_mean_ms": float(np.mean(values)),
            f"{prefix}_p50_ms": float(np.percentile(values, 50)),
            f"{prefix}_p95_ms": float(np.percentile(values, 95)),
        }

    summary: dict[str, float] = {}
    summary.update(summarize("preprocess_ms", "preprocess"))
    summary.update(summarize("inference_ms", "inference"))
    summary.update(summarize("postprocess_ms", "postprocess"))
    summary.update(summarize("total_pipeline_ms", "latency"))
    summary["fps_from_mean_latency"] = 1000.0 / summary["latency_mean_ms"]
    return summary


def evaluate_oiv7(
    *,
    model: Any,
    source: Path,
    items: list[ImageItem],
    classes: list[str],
    target_model_labels: dict[str, str],
    thresholds: dict[str, float],
    output_dir: Path,
    device: str,
    imgsz: int,
    confidence_floor: float,
    iou: float,
    max_det: int,
) -> dict[str, Any]:
    """Evaluate the unfiltered detector top-1 as a bbox-free verification output."""
    target_ids = resolve_target_class_ids(model.names, target_model_labels)
    rows = []
    for item in items:
        result = model.predict(
            source=str(item.source),
            verbose=False,
            device=device,
            imgsz=imgsz,
            conf=confidence_floor,
            iou=iou,
            max_det=max_det,
        )[0]
        rows.append(
            _prediction_row(
                item,
                source,
                result,
                target_ids,
                thresholds,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    class_summary = save_detector_visualizations(
        rows, classes, target_model_labels, output_dir
    )

    counts = Counter(row["true_label"] for row in rows)
    accepted_by_class = {
        label: sum(
            row["true_label"] == label and row["expected_label_pass"]
            for row in rows
        )
        for label in classes
    }
    acceptance_rates = {
        label: accepted_by_class[label] / counts[label] for label in classes
    }
    raw_top1_counts = Counter(row["model_top1_label"] for row in rows)
    model_path, model_size_mib = _model_size_mib(model)
    metrics = {
        "model": model_path or getattr(model, "ckpt_path", None) or "pretrained detector",
        "model_size_mib": model_size_mib,
        "mode": (
            f"unfiltered {len(model.names)}-class detection top-1; "
            "bbox not exported"
        ),
        "images": len(rows),
        "class_counts": dict(counts),
        "target_class_ids": target_ids,
        "thresholds": thresholds,
        "model_confidence_floor": confidence_floor,
        "top_k": 1,
        "expected_label_accept_count_by_class": accepted_by_class,
        "expected_label_accept_rate_by_class": acceptance_rates,
        "expected_label_accept_rate_macro": float(
            np.mean(list(acceptance_rates.values()))
        ),
        "service_reject_rate": sum(not row["expected_label_pass"] for row in rows)
        / len(rows),
        "model_top1_label_counts": dict(raw_top1_counts.most_common()),
        "model_top1_label_counts_by_true_class": {
            label: dict(
                Counter(
                    row["model_top1_label"]
                    for row in rows
                    if row["true_label"] == label
                ).most_common()
            )
            for label in classes
        },
        "no_detection_rate": sum(
            row["all_detection_count"] == 0 for row in rows
        )
        / len(rows),
        "class_summary": class_summary,
        "timing_scope": (
            "Ultralytics preprocess + inference + postprocess per image; "
            "model loading and source file discovery excluded"
        ),
        **_latency_summary(rows),
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
