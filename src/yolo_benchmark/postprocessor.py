"""Feature extraction and metrics for the trained RTMDet postprocessor."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .detectors import COCO_NAMES


TARGET_CLASSES = ("computer", "book", "other")


def truth_label(source_label: str) -> str:
    return source_label if source_label in TARGET_CLASSES[:2] else "other"


def extract_features(images: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Convert each image's RTMDet detections into fixed COCO score/count features."""
    indices = {name: index for index, name in enumerate(COCO_NAMES)}
    rows = []
    for image in images:
        max_scores = np.zeros(len(COCO_NAMES), dtype=np.float64)
        counts = np.zeros(len(COCO_NAMES), dtype=np.float64)
        for detection in image["detections"]:
            label = detection["label"]
            if label not in indices:
                raise ValueError(f"Unknown COCO label in detections: {label}")
            index = indices[label]
            max_scores[index] = max(max_scores[index], float(detection["score"]))
            counts[index] += 1
        rows.append(np.concatenate([max_scores, np.log1p(counts)]))
    if not rows:
        raise ValueError("No detections to featurize")
    names = [f"max_score:{name}" for name in COCO_NAMES]
    names += [f"log1p_count:{name}" for name in COCO_NAMES]
    return np.vstack(rows), names


def ranked_candidates(probabilities: np.ndarray, classes: list[str], k: int) -> list[list[str]]:
    if k not in (1, 2):
        raise ValueError("k must be 1 or 2")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Probability matrix does not match classes")
    order = np.argsort(-probabilities, axis=1)[:, :k]
    labels = np.asarray(classes)
    return [labels[row].tolist() for row in order]


def topk_summary(truth: list[str], candidates: list[list[str]]) -> list[dict]:
    rows = []
    truth_array = np.asarray(truth)
    for label in ("all", *TARGET_CLASSES):
        selected = np.ones(len(truth_array), dtype=bool) if label == "all" else truth_array == label
        indices = np.flatnonzero(selected)
        hits = sum(truth[index] in candidates[index] for index in indices)
        rows.append({"class": label, "images": len(indices), "hits": hits,
                     "rate": hits / len(indices) if len(indices) else None})
    return rows


def binary_metrics(truth: list[str], candidates: list[list[str]], label: str) -> dict:
    tp = sum(actual == label and label in predicted for actual, predicted in zip(truth, candidates))
    fp = sum(actual != label and label in predicted for actual, predicted in zip(truth, candidates))
    fn = sum(actual == label and label not in predicted for actual, predicted in zip(truth, candidates))
    tn = len(truth) - tp - fp - fn

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {"class": label, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": ratio(tp + tn, len(truth)), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "fpr": ratio(fp, fp + tn),
            "fnr": ratio(fn, tp + fn)}


def source_summary(source_labels: list[str], truth: list[str],
                   top1: list[list[str]], top2: list[list[str]]) -> list[dict]:
    output = []
    for source in sorted(Counter(source_labels)):
        indices = [index for index, value in enumerate(source_labels) if value == source]
        hits1 = sum(truth[index] in top1[index] for index in indices)
        hits2 = sum(truth[index] in top2[index] for index in indices)
        output.append({"source": source, "images": len(indices),
                       "top1": hits1 / len(indices), "top2": hits2 / len(indices)})
    return output
