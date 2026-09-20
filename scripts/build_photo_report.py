"""Build presentation artifacts from cached detections, without inference."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yolo_benchmark.reports import decide, evaluate, ratio, write_csv
from yolo_benchmark.common import write_json


def binary_metrics(rows, label):
    tp = sum(r["truth"] == label and label in r["candidates"] for r in rows)
    fp = sum(r["truth"] != label and label in r["candidates"] for r in rows)
    fn = sum(r["truth"] == label and label not in r["candidates"] for r in rows)
    tn = len(rows) - tp - fp - fn
    return {"class": label, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": ratio(tp + tn, len(rows)), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "fpr": ratio(fp, fp + tn), "fnr": ratio(fn, tp + fn)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/coco_model_suite/mvp-full-evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/photo-verification")
    args = parser.parse_args()
    snapshot = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    if snapshot["smoke_test"]:
        raise ValueError("Presentation requires a full evaluation")
    config = snapshot["config"]
    models, binary, hashes = [], [], {}
    for name in snapshot["selected_models"]:
        content = (args.run / name / "detections.json").read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        raw = json.loads(content)
        if {(r["path"], r["source_label"]) for r in raw["images"]} != {(r["path"], r["source_label"]) for r in snapshot["images"]} or len(raw["images"]) != len(snapshot["images"]):
            raise ValueError(f"Dataset mismatch: {name}")
        for k in ([1, 2] if name == "rtmdet-tiny" else [1]):
            rows = [{"truth": r["source_label"] if r["source_label"] in config["classes"] else "other",
                     "source_label": r["source_label"], **decide(r["detections"], {**config, "top_k": k})}
                    for r in raw["images"]]
            if k == 1:
                metrics = evaluate(rows, config)
                models.append({"model": name, **metrics})
            for label in config["classes"]:
                binary.append({"model": name, "top_k": k, **binary_metrics(rows, label)})
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "metrics.json", {"source_run": args.run.name, "images": len(snapshot["images"]),
        "source_counts": snapshot["evaluated_source_counts"], "detections_sha256": hashes,
        "criteria": config["report_criteria"], "top1": models, "category_verification": binary})
    write_csv(args.output / "category-verification.csv", binary)
    write_csv(args.output / "top1-models.csv", [{"model": m["model"], "accuracy": m["accuracy"],
        "macro_precision": m["macro"]["precision"], "macro_recall": m["macro"]["recall"],
        "macro_f1": m["macro"]["f1"], "other_to_target_fpr": m["image_false_accept_rate"]} for m in models])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    x = np.arange(len(models))
    for offset, key, title, color in [(-.26, "accuracy", "Accuracy", "#2563eb"),
                                     (0, "recall", "Macro recall", "#0d9488"),
                                     (.26, "precision", "Macro precision", "#a855f7")]:
        vals = [100 * (m[key] if key == "accuracy" else m["macro"][key]) for m in models]
        bars = axes[0].barh(x + offset, vals, .25, label=title, color=color)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[0].set(yticks=x, yticklabels=[m["model"] for m in models], xlim=(0, 112), xlabel="%", title="Top-1: 정확도와 클래스 균형 성능")
    for v, label, color in [(80, "최소 Acc 80%", "#64748b"), (90, "목표 Acc 90%", "#dc2626")]:
        axes[0].axvline(v, color=color, linestyle="--", label=label)
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.09), ncol=3)
    for offset, label, color in [(-.18, "computer", "#2563eb"), (.18, "book", "#f59e0b")]:
        vals = [100 * next(b["fpr"] for b in binary if b["model"] == m["model"] and b["top_k"] == 1 and b["class"] == label) for m in models]
        bars = axes[1].barh(x + offset, vals, .34, label=f"{label} 등록", color=color)
        axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(yticks=x, yticklabels=[m["model"] for m in models], xlim=(0, 10), xlabel="다른 클래스 사진 중 잘못 수락한 비율 (%)", title="Top-1: 등록 카테고리별 오수락률 FPR")
    axes[1].legend()
    fig.savefig(args.output / "top1-models.png", dpi=170)
    plt.close(fig)

    def matrix(ax, values, xticks, yticks, title):
        values = np.array(values)
        normalized = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set(xticks=range(len(xticks)), xticklabels=xticks, yticks=range(len(yticks)), yticklabels=yticks,
               xlabel="예측", ylabel="실제", title=title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i,j]}\n({normalized[i,j]:.1%})", ha="center", va="center",
                        color="white" if normalized[i,j] > .5 else "black")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), layout="constrained")
    for ax, m in zip(axes.flat, models):
        matrix(ax, m["confusion_matrix"], m["class_order"], m["class_order"], m["model"])
    axes.flat[-1].axis("off")
    fig.suptitle("Top-1 혼동행렬 · 건수와 행 정규화 비율 · 동일한 292장")
    fig.savefig(args.output / "top1-confusion.png", dpi=170)
    plt.close(fig)
    rt = [b for b in binary if b["model"] == "rtmdet-tiny"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
    for ax, b in zip(axes.flat, rt):
        matrix(ax, [[b["tp"], b["fn"]], [b["fp"], b["tn"]]], ["수락", "거절"], ["등록 클래스", "다른 클래스"], f"{b['class']} 등록 · top-{b['top_k']}")
    fig.suptitle("RTMDet-tiny: 등록 카테고리가 후보에 있으면 수락하는 정책의 오프라인 평가")
    fig.savefig(args.output / "rtmdet-confusion.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    keys = ["accuracy", "recall", "precision", "fpr"]
    for ax, label in zip(axes, config["classes"]):
        for offset, k, color in [(-.18, 1, "#2563eb"), (.18, 2, "#0d9488")]:
            b = next(b for b in rt if b["class"] == label and b["top_k"] == k)
            bars = ax.bar(np.arange(4) + offset, [100 * b[key] for key in keys], .35, label=f"top-{k}", color=color)
            ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set(xticks=range(4), xticklabels=["수락/거절 Acc", "Recall ↑", "Precision ↑", "FPR ↓"], ylim=(0, 115), title=f"{label} 등록", ylabel="%")
        ax.legend()
    fig.suptitle("후보 확대: 정답 회복과 오수락 증가를 함께 확인")
    fig.savefig(args.output / "rtmdet-tradeoff.png", dpi=170)
    plt.close(fig)
    print(json.dumps(rt, indent=2))


if __name__ == "__main__":
    main()
