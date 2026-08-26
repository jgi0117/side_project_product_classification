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
    accepted = scores >= threshold_values
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
                "expected_class_accepted": bool(
                    probabilities[target] >= threshold_values[target]
                ),
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
        # Primary accuracy: four independent one-vs-rest verification tasks.
        # A laptop is accepted from its own score, regardless of another
        # target class receiving a higher score.
        "accuracy": verification_accuracy,
        "auc_macro_ovr": float(auc),
        "verification_accuracy_macro": verification_accuracy,
        "verification_balanced_accuracy_macro": verification_balanced_accuracy,
        "open_set_top1_accuracy": float(open_set_top1_accuracy),
        "per_class_auc": per_class_auc,
        "verification_thresholds": thresholds,
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
        "accuracy",
        "auc_macro_ovr",
        "verification_accuracy_macro",
        "verification_balanced_accuracy_macro",
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
    frame["backend_label"] = frame["backend"].str.upper()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    palette = {"PYTORCH": "#4C78A8", "ONNX": "#F58518"}
    figure, axes_grid = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes_grid.flatten()
    specs = [
        ("accuracy", "Verification Accuracy", (0, 1), False, "%.3f"),
        ("auc_macro_ovr", "Macro OvR AUC", (0, 1), False, "%.3f"),
        ("latency_mean_ms", "Latency (ms/image)", None, True, "%.2f"),
        ("model_size_mb", "Model Size (MiB)", None, True, "%.2f"),
    ]
    for axis, (key, title, limits, lower_is_better, value_format) in zip(
        axes, specs
    ):
        sns.barplot(
            data=frame,
            x="model_label",
            y=key,
            hue="backend_label",
            palette=palette,
            errorbar=None,
            ax=axis,
        )
        axis.set_title(title, weight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("")
        if limits:
            axis.set_ylim(*limits)
        for container in axis.containers:
            axis.bar_label(container, fmt=value_format, padding=3, fontsize=9)
        if lower_is_better:
            axis.text(
                0.98, 0.96, "lower is better", transform=axis.transAxes,
                ha="right", va="top", fontsize=9, color="#666666",
            )
        if axis is not axes[0] and axis.legend_ is not None:
            axis.legend_.remove()
        axis.tick_params(axis="x", rotation=15)
    axes[0].legend(title="Backend", frameon=True, loc="lower left")
    sns.despine(fig=figure)
    backend_names = ", ".join(frame["backend_label"].drop_duplicates())
    figure.suptitle(
        f"YOLO Nano Zero-shot Verification — {backend_names}",
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
