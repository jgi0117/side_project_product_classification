from __future__ import annotations

import csv
import math
import platform
import statistics
import time
from collections import Counter
from contextlib import chdir
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from tqdm.auto import tqdm

from .common import write_json
from .data import ImageItem, discover_images


def resolve_device(requested: str) -> str | int:
    import torch

    if str(requested).lower() != "auto":
        value = str(requested)
        return int(value) if value.isdigit() else value
    return 0 if torch.cuda.is_available() else "cpu"


def _sync_cuda(device: str | int) -> None:
    import torch

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_imagenet_indices(
    names: dict[int, str] | list[str],
    classes: list[str],
    patterns: dict[str, list[str]],
) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    # Ultralytics labels use underscores (e.g. book_jacket); config uses spaces.
    normalized_names = {
        int(index): str(name).lower().replace("_", " ")
        for index, name in entries
    }
    indices: dict[str, list[int]] = {}
    matches: dict[str, list[str]] = {}
    for label in classes:
        label_patterns = [
            pattern.lower().replace("_", " ") for pattern in patterns[label]
        ]
        selected = [
            index
            for index, name in normalized_names.items()
            if any(pattern in name for pattern in label_patterns)
        ]
        if not selected:
            raise RuntimeError(
                f"ImageNet 클래스에서 {label!r} 패턴을 찾지 못했습니다: {patterns[label]}"
            )
        indices[label] = selected
        matches[label] = [normalized_names[index] for index in selected]
    return indices, matches


def _predict_scores(
    model: Any,
    model_name: str,
    paths: list[Path],
    classes: list[str],
    patterns: dict[str, list[str]],
    device: str | int,
    imgsz: int,
) -> tuple[np.ndarray, list[int], list[float], list[float], dict[str, list[str]]]:
    scores: list[list[float]] = []
    predictions: list[int] = []
    core_times: list[float] = []
    target_probability_masses: list[float] = []
    matched_names: dict[str, list[str]] | None = None
    class_indices: dict[str, list[int]] | None = None
    # Invoke both backends with exactly one image per call. Passing a list of
    # paths to an exported ONNX classifier can be assembled by Ultralytics as
    # one N-image tensor even when batch=1, while the exported graph has a
    # static batch dimension of 1.
    for path in tqdm(
        paths,
        desc=f"{Path(model_name).stem} scoring (not speed metric)",
        unit="image",
    ):
        batch_results = model.predict(
            source=str(path),
            imgsz=imgsz,
            batch=1,
            device=device,
            verbose=False,
        )
        if len(batch_results) != 1:
            raise RuntimeError(
                f"batch=1 추론 결과가 {len(batch_results)}개 반환됐습니다: {path}"
            )
        result = batch_results[0]
        if result.probs is None:
            raise RuntimeError("분류 확률이 없습니다. 반드시 *-cls.pt 모델을 사용하세요.")
        if class_indices is None:
            class_indices, matched_names = _resolve_imagenet_indices(
                result.names, classes, patterns
            )
        raw = result.probs.data.detach().cpu().numpy()
        aggregated = np.asarray(
            [float(raw[class_indices[label]].sum()) for label in classes]
        )
        total = float(aggregated.sum())
        target_probability_masses.append(total)
        # Keep original ImageNet probability mass. Do not force the output to
        # sum to one across only the four target classes.
        scores.append(aggregated.tolist())
        predictions.append(int(np.argmax(aggregated)))
        if result.speed and "inference" in result.speed:
            core_times.append(float(result.speed["inference"]))
    assert matched_names is not None
    return (
        np.asarray(scores),
        predictions,
        core_times,
        target_probability_masses,
        matched_names,
    )


def _primary_label_operating_metrics(
    targets: list[int] | np.ndarray,
    accepted: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Metrics that remain valid when folder labels are not exhaustive.

    A folder label is treated as one object known to be present. Other accepted
    labels are reported as additional activations, not false positives, because
    another configured object may also be visible in the same image.
    """
    target_array = np.asarray(targets, dtype=int)
    accepted_array = np.asarray(accepted, dtype=bool)
    expected_shape = (len(target_array), len(classes))
    if accepted_array.shape != expected_shape:
        raise ValueError(
            f"accepted shape must be {expected_shape}, got {accepted_array.shape}"
        )

    accepted_counts = accepted_array.sum(axis=1)
    expected_accepted = accepted_array[np.arange(len(target_array)), target_array]
    per_class: dict[str, dict[str, float | int]] = {}
    target_accept_rates: list[float] = []
    additional_activation_rates: list[float] = []
    for index, name in enumerate(classes):
        rows = target_array == index
        image_count = int(rows.sum())
        row_accepts = accepted_array[rows]
        target_accept_rate = float(row_accepts[:, index].mean())
        additional_mask = row_accepts.copy()
        additional_mask[:, index] = False
        additional_activation_rate = float(np.any(additional_mask, axis=1).mean())
        target_accept_rates.append(target_accept_rate)
        additional_activation_rates.append(additional_activation_rate)
        per_class[name] = {
            "images": image_count,
            "expected_label_accepts": int(row_accepts[:, index].sum()),
            "expected_label_rejects": int((~row_accepts[:, index]).sum()),
            "expected_label_accept_rate": target_accept_rate,
            "expected_label_reject_rate": float(1.0 - target_accept_rate),
            "additional_label_activation_rate": additional_activation_rate,
        }

    return {
        "expected_label_accept_rate_macro": float(
            statistics.fmean(target_accept_rates)
        ),
        "expected_label_reject_rate_macro": float(
            statistics.fmean(1.0 - rate for rate in target_accept_rates)
        ),
        "additional_label_activation_rate_macro": float(
            statistics.fmean(additional_activation_rates)
        ),
        "multiple_label_rate": float(np.mean(accepted_counts > 1)),
        "no_label_rate": float(np.mean(accepted_counts == 0)),
        "expected_label_accepted_overall": float(expected_accepted.mean()),
        "per_class_primary_label_metrics": per_class,
        "label_semantics": "folder_label_is_known_present_but_not_exhaustive",
        "false_accept_rate_available": False,
    }


def _apply_top_k_thresholds(
    scores: np.ndarray,
    threshold_values: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Accept threshold-passing labels only when they rank within top-k."""
    score_array = np.asarray(scores, dtype=float)
    thresholds = np.asarray(threshold_values, dtype=float)
    if score_array.ndim != 2 or thresholds.shape != (score_array.shape[1],):
        raise ValueError("scores and threshold_values have incompatible shapes")
    if not 1 <= top_k <= score_array.shape[1]:
        raise ValueError(f"top_k must be between 1 and {score_array.shape[1]}")

    ranked_indices = np.argsort(-score_array, axis=1)[:, :top_k]
    accepted = np.zeros_like(score_array, dtype=bool)
    rows = np.arange(score_array.shape[0])[:, None]
    accepted[rows, ranked_indices] = (
        score_array[rows, ranked_indices] >= thresholds[ranked_indices]
    )
    return accepted


def _measure_latency(
    model: Any,
    model_name: str,
    paths: list[Path],
    device: str | int,
    imgsz: int,
    warmup: int,
    repeats: int,
) -> list[float]:
    import cv2

    warm_image = cv2.imread(str(paths[0]))
    if warm_image is None:
        raise RuntimeError(f"속도 측정용 이미지 디코딩에 실패했습니다: {paths[0]}")
    for _ in range(warmup):
        model.predict(warm_image, imgsz=imgsz, batch=1, device=device, verbose=False)
    _sync_cuda(device)
    latencies: list[float] = []
    measurements = (
        path for _ in range(repeats) for path in paths
    )
    for path in tqdm(
        measurements,
        total=repeats * len(paths),
        desc=f"{Path(model_name).stem} speed test",
        unit="image",
    ):
        # Decode outside the timer and keep only one image in RAM at a time.
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"속도 측정용 이미지 디코딩에 실패했습니다: {path}")
        _sync_cuda(device)
        start = time.perf_counter()
        model.predict(image, imgsz=imgsz, batch=1, device=device, verbose=False)
        _sync_cuda(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def _load_backend_model(
    model_name: str,
    model_dir: Path,
    backend: str,
    imgsz: int,
    onnx_simplify: bool,
) -> tuple[Any, Path]:
    from ultralytics import YOLO

    model_dir.mkdir(parents=True, exist_ok=True)
    supplied_path = Path(model_name)
    if supplied_path.is_absolute() or supplied_path.exists():
        pt_checkpoint = supplied_path.resolve()
    else:
        pt_checkpoint = (model_dir / supplied_path.name).resolve()
        if not pt_checkpoint.exists():
            with chdir(model_dir):
                YOLO(supplied_path.name)
        if not pt_checkpoint.exists():
            raise FileNotFoundError(f"체크포인트를 찾지 못했습니다: {pt_checkpoint}")

    if backend == "pytorch":
        return YOLO(str(pt_checkpoint)), pt_checkpoint
    if backend != "onnx":
        raise ValueError(f"지원하지 않는 backend입니다: {backend}")

    onnx_checkpoint = pt_checkpoint.with_suffix(".onnx")
    if not onnx_checkpoint.exists():
        exported = YOLO(str(pt_checkpoint)).export(
            format="onnx",
            imgsz=imgsz,
            simplify=onnx_simplify,
            device="cpu",
        )
        exported_path = Path(exported).resolve()
        if exported_path != onnx_checkpoint and exported_path.exists():
            onnx_checkpoint = exported_path
    if not onnx_checkpoint.exists():
        raise FileNotFoundError(f"ONNX export 결과가 없습니다: {onnx_checkpoint}")
    return YOLO(str(onnx_checkpoint), task="classify"), onnx_checkpoint


def evaluate_zero_shot(
    model_name: str,
    backend: str,
    source: Path,
    model_dir: Path,
    output_dir: Path,
    classes: list[str],
    patterns: dict[str, list[str]],
    device: str | int,
    imgsz: int,
    speed_warmup: int,
    speed_repeats: int,
    thresholds: dict[str, float],
    top_k: int,
    onnx_simplify: bool,
    items: list[ImageItem] | None = None,
    invalid_images: int = 0,
) -> dict[str, Any]:
    import torch
    import ultralytics

    if items is None:
        items, invalid_images = discover_images(source, classes)
    paths = [item.source for item in items]
    targets = [classes.index(item.label) for item in items]
    model, checkpoint = _load_backend_model(
        model_name, model_dir, backend, imgsz, onnx_simplify
    )
    run_label = f"{Path(model_name).stem}-{backend}"
    scores, predictions, core_times, target_masses, matched_names = _predict_scores(
        model, run_label, paths, classes, patterns, device, imgsz
    )
    threshold_values = np.asarray([float(thresholds[name]) for name in classes])
    accepted = _apply_top_k_thresholds(scores, threshold_values, top_k)
    operating_metrics = _primary_label_operating_metrics(targets, accepted, classes)
    open_predictions = [
        prediction
        if scores[row, prediction] >= threshold_values[prediction]
        else len(classes)
        for row, prediction in enumerate(predictions)
    ]
    open_set_top1_accuracy = accuracy_score(targets, open_predictions)
    binary_accuracies = []
    binary_balanced_accuracies = []
    per_class_auc = {}
    for index, name in enumerate(classes):
        truth = np.asarray(targets) == index
        class_accepted = accepted[:, index]
        binary_accuracies.append(accuracy_score(truth, class_accepted))
        binary_balanced_accuracies.append(
            balanced_accuracy_score(truth, class_accepted)
        )
        per_class_auc[name] = float(roc_auc_score(truth, scores[:, index]))
    auc = float(statistics.fmean(per_class_auc.values()))
    verification_accuracy = float(statistics.fmean(binary_accuracies))
    verification_balanced_accuracy = float(
        statistics.fmean(binary_balanced_accuracies)
    )
    matrix = confusion_matrix(
        targets, open_predictions, labels=list(range(len(classes) + 1))
    )
    latencies = _measure_latency(
        model, run_label, paths, device, imgsz, speed_warmup, speed_repeats
    )
    mean_latency = statistics.fmean(latencies)

    # Keep backend artifacts separate so each backend has an independent
    # three-model comparison under outputs/benchmark/<backend>/.
    run_name = Path(model_name).stem
    model_output = output_dir / backend / run_name
    model_output.mkdir(parents=True, exist_ok=True)
    with (model_output / "predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "path",
            "true_label",
            "predicted_label",
            "accepted_labels",
            "expected_class_score",
            "expected_class_accepted",
            "target_probability_mass",
            *[f"score_{name}" for name in classes],
            *[f"accepted_{name}" for name in classes],
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, (path, target, target_mass, probabilities) in enumerate(
            zip(paths, targets, target_masses, scores)
        ):
            open_prediction = open_predictions[row_index]
            row = {
                "path": str(path),
                "true_label": classes[target],
                "predicted_label": (
                    classes[open_prediction]
                    if open_prediction < len(classes)
                    else "unknown"
                ),
                "accepted_labels": ";".join(
                    name
                    for index, name in enumerate(classes)
                    if accepted[row_index, index]
                ) or "unknown",
                "expected_class_score": float(probabilities[target]),
                "expected_class_accepted": bool(accepted[row_index, target]),
                "target_probability_mass": target_mass,
            }
            row.update(
                {
                    f"score_{name}": float(probabilities[index])
                    for index, name in enumerate(classes)
                }
            )
            row.update(
                {
                    f"accepted_{name}": bool(accepted[row_index, index])
                    for index, name in enumerate(classes)
                }
            )
            writer.writerow(row)

    result = {
        "model": model_name,
        "backend": backend,
        "checkpoint": str(checkpoint),
        "model_size_mb": float(checkpoint.stat().st_size / (1024**2)),
        "mode": "zero_shot_imagenet_open_set_verification",
        # Legacy single-label proxy metrics are retained for compatibility.
        # They must not be interpreted as false-accept measurements because a
        # folder label does not prove that other configured objects are absent.
        "accuracy": verification_accuracy,
        "auc_macro_ovr": float(auc),
        "verification_accuracy_macro": verification_accuracy,
        "verification_balanced_accuracy_macro": verification_balanced_accuracy,
        **operating_metrics,
        "open_set_top1_accuracy": float(open_set_top1_accuracy),
        "per_class_auc": per_class_auc,
        "verification_thresholds": thresholds,
        "top_k": top_k,
        "latency_mean_ms": float(mean_latency),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "fps": float(1000.0 / mean_latency),
        "core_inference_mean_ms": (
            float(statistics.fmean(core_times)) if core_times else math.nan
        ),
        "target_probability_mass_mean": float(statistics.fmean(target_masses)),
        "target_probability_mass_p50": float(np.percentile(target_masses, 50)),
        "images": len(paths),
        "invalid_images_skipped": invalid_images,
        "class_counts": dict(Counter(item.label for item in items)),
        "speed_samples": len(latencies),
        "confusion_matrix": matrix.tolist(),
        "class_order": [*classes, "unknown"],
        "matched_imagenet_classes": matched_names,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device != "cpu" and torch.cuda.is_available()
            else platform.processor() or "CPU"
        ),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
    }
    write_json(model_output / "metrics.json", result)
    return result


def save_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "backend",
        "checkpoint",
        "model_size_mb",
        "mode",
        "top_k",
        "accuracy",
        "auc_macro_ovr",
        "verification_accuracy_macro",
        "verification_balanced_accuracy_macro",
        "expected_label_accept_rate_macro",
        "expected_label_reject_rate_macro",
        "additional_label_activation_rate_macro",
        "multiple_label_rate",
        "no_label_rate",
        "open_set_top1_accuracy",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "fps",
        "core_inference_mean_ms",
        "target_probability_mass_mean",
        "images",
        "invalid_images_skipped",
        "speed_samples",
        "device",
        "device_name",
        "ultralytics_version",
        "torch_version",
    ]
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})
    write_json(output_dir / "summary.json", results)

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    frame = pd.DataFrame(results)
    frame["model_label"] = frame["model"].map(lambda value: Path(value).stem)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    figure, axes_grid = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes_grid.flatten()
    specs = [
        ("expected_label_accept_rate_macro", "Expected Label Accept Rate ↑", (0, 1), "%.1f%%", 100),
        ("expected_label_reject_rate_macro", "Expected Label Reject Rate ↓", (0, 1), "%.1f%%", 100),
        ("additional_label_activation_rate_macro", "Additional Label Activation", (0, 1), "%.1f%%", 100),
        ("multiple_label_rate", "Multiple-label Rate", (0, 1), "%.1f%%", 100),
        ("latency_mean_ms", "Latency (ms/image) ↓", None, "%.2f", 1),
        ("model_size_mb", "Model Size (MiB) ↓", None, "%.2f", 1),
    ]
    for axis, (key, title, limits, value_format, display_scale) in zip(axes, specs):
        plot_frame = frame.copy()
        plot_frame["display_value"] = plot_frame[key] * display_scale
        sns.barplot(
            data=plot_frame,
            x="model_label",
            y="display_value",
            color="#4C78A8",
            errorbar=None,
            ax=axis,
        )
        axis.set_title(title, weight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("")
        if limits:
            axis.set_ylim(limits[0] * display_scale, limits[1] * display_scale)
        for container in axis.containers:
            axis.bar_label(container, fmt=value_format, padding=3, fontsize=9)
        axis.tick_params(axis="x", rotation=15)
    sns.despine(fig=figure)
    backend_names = ", ".join(frame["backend"].str.upper().drop_duplicates())
    figure.suptitle(
        f"Camera Object Verification — {backend_names}",
        fontsize=17,
        weight="bold",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "comparison.png", dpi=220, bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    class_order = list(results[0]["per_class_primary_label_metrics"])
    model_order = frame["model_label"].tolist()
    target_accept_matrix = np.asarray(
        [
            [
                result["per_class_primary_label_metrics"][name][
                    "expected_label_accept_rate"
                ]
                for name in class_order
            ]
            for result in results
        ]
    )
    additional_activation_matrix = np.asarray(
        [
            [
                result["per_class_primary_label_metrics"][name][
                    "additional_label_activation_rate"
                ]
                for name in class_order
            ]
            for result in results
        ]
    )
    class_figure, class_axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, matrix, title, color in (
        (class_axes[0], target_accept_matrix, "Expected Label Accept Rate ↑", "Blues"),
        (
            class_axes[1],
            additional_activation_matrix,
            "Additional Label Activation (not confirmed false)",
            "Oranges",
        ),
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1%",
            vmin=0,
            vmax=1,
            cmap=color,
            xticklabels=class_order,
            yticklabels=model_order,
            cbar=False,
            linewidths=0.5,
            ax=axis,
        )
        axis.set_title(title, weight="bold")
        axis.set_xlabel("Folder label (known-present object)")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=0)
        axis.tick_params(axis="y", rotation=0)
    class_figure.suptitle(
        f"Per-class Camera Verification — {backend_names}",
        fontsize=16,
        weight="bold",
        y=1.02,
    )
    class_figure.tight_layout()
    class_figure.savefig(
        output_dir / "verification_by_class.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(class_figure)
