from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


COCO_CLASSES = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
    "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
    "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,"
    "apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
    "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,"
    "microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,"
    "hair drier,toothbrush"
).split(",")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_cv_image(path: str | Path):
    """Read an OpenCV image without losing Unicode Windows paths."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {path}")
    return image


def artifact_size_mib(path: Path) -> float:
    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        suffixes = {".pdmodel", ".pdiparams", ".json", ".yml", ".yaml"}
        size = sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in suffixes
        )
    else:
        raise FileNotFoundError(f"모델 파일을 찾지 못했습니다: {path}")
    return size / (1024**2)


def write_adapter_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    framework: str,
    artifact: Path,
    timing_scope: str,
    top_k: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_fields = [
        field
        for rank in range(1, top_k + 1)
        for field in (
            f"model_top{rank}_class_id",
            f"model_top{rank}_label",
            f"model_top{rank}_score",
        )
    ]
    fieldnames = [
        "file",
        "true_label",
        *rank_fields,
        "all_detection_count",
        "unique_detection_class_count",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "total_pipeline_ms",
    ]
    with (output_dir / "raw_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    totals = [float(row["total_pipeline_ms"]) for row in rows]
    metadata = {
        "model_id": model_id,
        "framework": framework,
        "artifact": str(artifact.resolve()),
        "model_size_mib": artifact_size_mib(artifact),
        "images": len(rows),
        "top_k": top_k,
        "timing_scope": timing_scope,
        "latency_mean_ms": statistics.fmean(totals),
        "latency_p50_ms": statistics.median(totals),
        "latency_p95_ms": sorted(totals)[max(0, int(len(totals) * 0.95) - 1)],
    }
    with (output_dir / "adapter_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def topk_row(
    item: dict[str, str],
    detections: list[tuple[int, str, float]],
    timings: tuple[float, float, float],
    top_k: int,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    # 같은 클래스의 여러 bbox가 top-k를 독점하지 않도록 클래스별 최고 점수만 사용합니다.
    best_by_class: dict[int, tuple[int, str, float]] = {}
    for detection in detections:
        class_id = detection[0]
        if class_id not in best_by_class or detection[2] > best_by_class[class_id][2]:
            best_by_class[class_id] = detection
    ranked = sorted(best_by_class.values(), key=lambda value: value[2], reverse=True)
    preprocess_ms, inference_ms, postprocess_ms = timings
    row: dict[str, Any] = {
        "file": item["file"],
        "true_label": item["true_label"],
        "all_detection_count": len(detections),
        "unique_detection_class_count": len(best_by_class),
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_pipeline_ms": preprocess_ms + inference_ms + postprocess_ms,
    }
    for rank in range(1, top_k + 1):
        class_id, label, score = (
            ranked[rank - 1] if rank <= len(ranked) else (-1, "unknown", 0.0)
        )
        row[f"model_top{rank}_class_id"] = class_id
        row[f"model_top{rank}_label"] = label
        row[f"model_top{rank}_score"] = score
    return row


def top1_row(
    item: dict[str, str],
    detections: list[tuple[int, str, float]],
    timings: tuple[float, float, float],
) -> dict[str, Any]:
    return topk_row(item, detections, timings, top_k=1)
