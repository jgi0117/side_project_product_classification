from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import write_json
from .data import ImageItem


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
                f"체크포인트에서 OIV7 클래스 {model_label!r}를 찾지 못했습니다."
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
        "oiv7_top1_class_id": raw_class_id,
        "oiv7_top1_label": raw_label,
        "oiv7_top1_score": raw_score,
        "decision": "pass" if expected_label_pass else "reject",
        "expected_label_pass": expected_label_pass,
        "all_detection_count": len(all_detections),
    }
    return row


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
    """Evaluate the unfiltered OIV7 top-1 as a bbox-free verification output."""
    target_ids = resolve_target_class_ids(model.names, target_model_labels)
    paths: Iterable[str] = (str(item.source) for item in items)
    predictions = model.predict(
        source=list(paths),
        stream=True,
        verbose=False,
        device=device,
        imgsz=imgsz,
        conf=confidence_floor,
        iou=iou,
        max_det=max_det,
    )

    rows = [
        _prediction_row(
            item,
            source,
            result,
            target_ids,
            thresholds,
        )
        for item, result in zip(items, predictions, strict=True)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

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
    raw_top1_counts = Counter(row["oiv7_top1_label"] for row in rows)
    metrics = {
        "model": getattr(model, "ckpt_path", None) or "yolov8n-oiv7.pt",
        "mode": "unfiltered OIV7 601-class detection top-1; bbox not exported",
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
        "oiv7_top1_label_counts": dict(raw_top1_counts.most_common()),
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
