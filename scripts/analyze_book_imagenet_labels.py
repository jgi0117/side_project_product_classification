from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.benchmark import _resolve_imagenet_indices  # noqa: E402
from yolo_benchmark.common import DEFAULT_CONFIG, load_config, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect original ImageNet top-k labels for book-folder images"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    from ultralytics import YOLO

    config = load_config(args.config)
    output_dir = config["output_dir"] / "pytorch"
    records: list[dict[str, object]] = []
    top1_counts: dict[str, Counter[str]] = {}
    mapped_book_labels: set[str] = set()

    for model_name in config["models"]:
        run_name = Path(model_name).stem
        prediction_path = output_dir / run_name / "predictions.csv"
        with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
            book_rows = [
                row for row in csv.DictReader(handle) if row["true_label"] == "book"
            ]
        checkpoint = config["model_dir"] / Path(model_name).name
        model = YOLO(str(checkpoint))
        model_top1: Counter[str] = Counter()
        for row in book_rows:
            result = model.predict(
                row["path"],
                imgsz=int(config["imgsz"]),
                batch=1,
                device=args.device,
                verbose=False,
            )[0]
            raw = result.probs.data.detach().cpu().numpy()
            class_indices, _ = _resolve_imagenet_indices(
                result.names,
                list(config["classes"]),
                config["imagenet_class_patterns"],
            )
            book_indices = set(class_indices["book"])
            ranked = np.argsort(-raw)[: min(args.top_k, len(raw))]
            for rank, index in enumerate(ranked, start=1):
                label = str(result.names[int(index)])
                is_book_mapping = int(index) in book_indices
                if is_book_mapping:
                    mapped_book_labels.add(label)
                records.append(
                    {
                        "model": run_name,
                        "file": Path(row["path"]).name,
                        "rank": rank,
                        "imagenet_index": int(index),
                        "imagenet_label": label,
                        "probability": float(raw[index]),
                        "mapped_to_book": is_book_mapping,
                    }
                )
                if rank == 1:
                    model_top1[label] += 1
        top1_counts[run_name] = model_top1

    csv_path = output_dir / "book_imagenet_topk.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    model_order = [Path(name).stem for name in config["models"]]
    label_order = sorted(
        {label for counts in top1_counts.values() for label in counts},
        key=lambda label: (
            -sum(top1_counts[model][label] for model in model_order),
            label,
        ),
    )
    summary = {
        model: dict(counts.most_common()) for model, counts in top1_counts.items()
    }
    write_json(output_dir / "book_imagenet_top1_summary.json", summary)

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="talk", font_scale=0.8)
    plot_labels = [
        label
        for label in label_order
        if sum(top1_counts[model][label] for model in model_order) >= 2
        or label in mapped_book_labels
    ]
    omitted_labels = [label for label in label_order if label not in plot_labels]
    matrix_rows = [
        [top1_counts[model][label] for model in model_order]
        for label in plot_labels
    ]
    if omitted_labels:
        plot_labels.append("other single-occurrence labels")
        matrix_rows.append(
            [
                sum(top1_counts[model][label] for label in omitted_labels)
                for model in model_order
            ]
        )
    matrix = np.asarray(matrix_rows)
    display_labels = [
        f"[book map] {label}" if label in mapped_book_labels else label
        for label in plot_labels
    ]
    figure_height = max(6.0, 0.48 * len(plot_labels) + 2.0)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0,
        cbar=False,
        linewidths=0.5,
        xticklabels=model_order,
        yticklabels=display_labels,
        ax=axis,
    )
    axis.set_title(
        "Book Images: Original ImageNet Top-1 Labels",
        weight="bold",
        pad=16,
    )
    axis.set_xlabel("Model")
    axis.set_ylabel("ImageNet-1K top-1 label ([book map] configured as book)")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(
        output_dir / "book_imagenet_top1_labels.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(f"Saved: {csv_path.resolve()}")
    print(f"Saved: {(output_dir / 'book_imagenet_top1_labels.png').resolve()}")


if __name__ == "__main__":
    main()
