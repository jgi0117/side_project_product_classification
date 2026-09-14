"""Image-level MVP decisions and reproducible reports from cached detections."""
from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .common import write_json


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def decide(detections, config):
    """Max confidence per group; highest threshold-passing group wins.

    Detector confidences are not exclusive probabilities, so laptop and TV
    scores must not be added (nor renormalized). No detection maps to other.
    """
    reverse = {raw: group for group, names in config["coco_class_mapping"].items()
               for raw in names}
    labels = [*config["classes"], "other"]
    scores = dict.fromkeys(labels, 0.0)
    for det in detections:
        group = reverse.get(det["label"], "other")
        scores[group] = max(scores[group], float(det["score"]))
    thresholds = {**config["verification_thresholds"], "other": config["other_threshold"]}
    eligible = [label for label in labels if scores[label] > 0 and scores[label] >= thresholds[label]]
    predicted = max(eligible, key=lambda label: scores[label]) if eligible else "other"
    raw = max(detections, key=lambda det: det["score"], default=None)
    return {"prediction": predicted, "scores": scores,
            "raw_top1": raw["label"] if raw else "no_detection",
            "raw_top1_confidence": float(raw["score"]) if raw else 0.0,
            "decision_reason": "highest_passing_group" if eligible else "no_passing_detection"}


def evaluate(rows, config):
    labels = [*config["classes"], "other"]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for row in rows:
        matrix[labels.index(row["truth"]), labels.index(row["prediction"])] += 1
    total = int(matrix.sum())
    if not total:
        raise ValueError("No predictions to evaluate")
    per_class = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fn = int(matrix[index].sum()) - tp
        fp = int(matrix[:, index].sum()) - tp
        tn = total - tp - fn - fp
        per_class.append({"class": label, "support": tp + fn, "tp": tp, "fp": fp,
                          "fn": fn, "tn": tn, "precision": ratio(tp, tp + fp),
                          "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
                          "false_accept_rate": ratio(fp, fp + tn),
                          "false_reject_rate": ratio(fn, tp + fn)})
    accuracy = float(np.trace(matrix) / total)
    # A wrong MVP class is an overall error, but is not a binary rejection.
    other = labels.index("other")
    negative_count = int(matrix[other].sum())
    positive_count = total - negative_count
    macro = {}
    for metric in ("precision", "recall", "f1"):
        values = [row[metric] for row in per_class]
        macro[metric] = sum(values) / len(values) if all(v is not None for v in values) else None
    criteria = config["report_criteria"]
    checks = [
        {"item": "목표 전체 판별 정확도", "scope": "image", "value": accuracy,
         "operator": ">=", "threshold": criteria["target_accuracy"],
         "status": "PASS" if accuracy >= criteria["target_accuracy"] else "FAIL"},
        {"item": "최소 허용 정확도", "scope": "image", "value": accuracy,
         "operator": ">=", "threshold": criteria["minimum_accuracy"],
         "status": "PASS" if accuracy >= criteria["minimum_accuracy"] else "FAIL"},
        {"item": "세션 기준 수락 오류", "scope": "session", "value": None,
         "operator": "<=", "threshold": criteria["session_false_accept_rate"], "status": "N/A"},
        {"item": "세션 기준 거절 오류", "scope": "session", "value": None,
         "operator": "<=", "threshold": criteria["session_false_reject_rate"], "status": "N/A"},
    ]
    return {"images": total, "accuracy": accuracy, "macro": macro, "per_class": per_class,
            "class_order": labels, "confusion_matrix": matrix.tolist(), "criteria": checks,
            "image_false_accept_rate": ratio(int(matrix[other, :other].sum()), negative_count),
            "image_false_reject_rate": ratio(int(matrix[:other, other].sum()), positive_count),
            "negative_images": negative_count, "positive_images": positive_count,
            "source_counts": dict(Counter(row["source_label"] for row in rows)),
            "session_false_accept_rate": None, "session_false_reject_rate": None,
            "annotation_basis": "single primary folder label; other folders assumed negative",
            "session_status": "No session annotations; image rates are not session rates"}


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def html_table(headers, rows):
    def cells(values, tag):
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values) + "</tr>"
    return "<div class='table-wrap'><table>" + cells(headers, "th") + "".join(
        cells(row, "td") for row in rows) + "</table></div>"


def render_model(raw, config, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in raw["images"]:
        decision = decide(item["detections"], config)
        rows.append({**item, **decision,
                     "truth": item["source_label"] if item["source_label"] in config["classes"] else "other"})
    metrics = evaluate(rows, config)
    metrics.update({"model": raw["model"], "runtime": raw["runtime"],
                    "mapping": config["coco_class_mapping"],
                    "thresholds": config["verification_thresholds"],
                    "other_threshold": config["other_threshold"]})
    run_path = output.parent / "run.json"
    smoke = run_path.is_file() and json.loads(run_path.read_text(encoding="utf-8"))["smoke_test"]
    metrics["smoke_test"] = bool(smoke)
    if smoke:
        for check in metrics["criteria"]:
            if check["scope"] == "image":
                check["status"] = "SMOKE_ONLY"
    write_json(output / "metrics.json", metrics)
    flat_rows = [{"path": row["path"], "source_label": row["source_label"],
                  "true_label": row["truth"], "prediction": row["prediction"],
                  "correct": row["truth"] == row["prediction"],
                  "raw_top1": row["raw_top1"], "raw_top1_confidence": row["raw_top1_confidence"],
                  "decision_reason": row["decision_reason"],
                  **{f"score_{k}": v for k, v in row["scores"].items()},
                  "detections": json.dumps(row["detections"], ensure_ascii=False)} for row in rows]
    write_csv(output / "predictions.csv", flat_rows)
    write_csv(output / "per_class.csv", metrics["per_class"])
    write_csv(output / "criteria.csv", metrics["criteria"])
    labels = metrics["class_order"]
    matrix = np.array(metrics["confusion_matrix"])
    write_csv(output / "confusion_matrix.csv", [
        {"true_label": label, **dict(zip(labels, matrix[index].tolist()))}
        for index, label in enumerate(labels)])
    write_csv(output / "confusion_matrix_normalized.csv", [
        {"true_label": label, **{name: ratio(int(matrix[index, j]), int(matrix[index].sum()))
                                for j, name in enumerate(labels)}}
        for index, label in enumerate(labels)])
    source_rows = []
    for source in sorted(metrics["source_counts"]):
        subset = [row for row in rows if row["source_label"] == source]
        counts = Counter(row["prediction"] for row in subset)
        source_rows.append({"source_label": source, "images": len(subset),
                            "accuracy": sum(row["truth"] == row["prediction"] for row in subset) / len(subset),
                            **{label: counts[label] for label in labels}})
    write_csv(output / "by_source.csv", source_rows)
    raw_counts = Counter((row["source_label"], row["raw_top1"]) for row in rows)
    write_csv(output / "raw_coco_outcomes.csv", [
        {"source_label": source, "raw_coco_prediction": predicted, "count": count}
        for (source, predicted), count in sorted(raw_counts.items())])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, values, title in zip(axes, [matrix, normalized], ["Counts", "Row-normalized recall"]):
        ax.imshow(values, cmap="Blues", vmin=0, vmax=max(1, values.max()))
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
               yticklabels=labels, xlabel="Predicted", ylabel="True", title=title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = str(matrix[i, j]) if ax is axes[0] else (percent(values[i, j]) if support[i, 0] else "N/A")
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if values[i, j] > values.max() / 2 else "black")
    fig.suptitle(raw["model"])
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    lines = [f"# {raw['model']} — COCO MVP report", "",
             f"Images: {metrics['images']} | Accuracy: {percent(metrics['accuracy'])}", "",
             "| Class | Images | Precision | Recall | F1 | FPR | FNR |", "|---|---:|---:|---:|---:|---:|---:|"]
    if smoke:
        lines[2:2] = ["샘플 실행 검증용 결과입니다. 전체 데이터 성능이나 기준 충족의 근거로 사용하지 않습니다.", ""]
    for row in metrics["per_class"]:
        lines.append(f"| {row['class']} | {row['support']} | " + " | ".join(
            percent(row[key]) for key in ("precision", "recall", "f1", "false_accept_rate", "false_reject_rate")) + " |")
    lines += ["", "| 기준 | 측정값 | 요구값 | 판정 |", "|---|---:|---:|---|"]
    for check in metrics["criteria"]:
        lines.append(f"| {check['item']} | {percent(check['value'])} | {check['operator']} {percent(check['threshold'])} | {check['status']} |")
    lines += ["", f"이미지 기준 수락 오류율(other → MVP): {percent(metrics['image_false_accept_rate'])}",
              f"이미지 기준 거절 오류율(MVP → other): {percent(metrics['image_false_reject_rate'])}", "",
              "세션 정보가 없어 세션 기준 오류율은 N/A입니다. 이미지 오류율로 세션 기준 통과 여부를 판단하지 않습니다.",
              "서로 다른 MVP 클래스 간 오판은 전체 정확도와 클래스별 FPR/FNR에 반영됩니다.", "",
              "폴더명을 단일 주 객체 정답으로 가정한 이미지 분류 지표입니다. 바운딩 박스 탐지 mAP가 아닙니다.",
              "다른 카테고리 사진에 실제 book/computer가 함께 있으면 수락 오류가 과대 집계될 수 있습니다.",
              "정확한 운영 평가에는 복수 객체 및 세션 정답이 필요합니다. 분모가 0인 지표는 N/A입니다.", "",
              "![Confusion matrix](confusion_matrix.png)", "",
              "원본 COCO 예측: raw_coco_outcomes.csv · 출처별 결과: by_source.csv · 이미지별 결과: predictions.csv", "",
              "Runtime (decoded-image preprocessing + forward + postprocessing, batch=1; excludes disk I/O):", "",
              "```json", json.dumps(raw["runtime"], ensure_ascii=False, indent=2), "```", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def render_summary(results, errors, output):
    write_json(output / "summary.json", {"models": results, "errors": errors})
    columns = ["model", "images", "accuracy", "macro_precision", "macro_recall", "macro_f1",
               "image_false_accept_rate", "image_false_reject_rate", "latency_mean_ms", "fps", "model_size_mib"]
    rows = [{"model": m["model"], "images": m["images"], "accuracy": m["accuracy"],
             **{f"macro_{k}": v for k, v in m["macro"].items()},
             "image_false_accept_rate": m["image_false_accept_rate"],
             "image_false_reject_rate": m["image_false_reject_rate"],
             **{key: m["runtime"][key] for key in columns[-3:]}} for m in results]
    write_csv(output / "summary.csv", rows, columns)
    lines = ["# COCO MVP 모델 비교", "", "| Model | Images | Accuracy | Macro precision | Macro recall | Macro F1 | Image FPR | Image FNR | ms | MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| [{row['model']}]({row['model']}/report.md) | {row['images']} | " + " | ".join(
            percent(row[key]) for key in columns[2:8]) + f" | {row['latency_mean_ms']:.2f} | {row['model_size_mib']:.2f} |")
    lines += ["", "목표 정확도 ≥90%, 최소 허용 정확도 ≥80%. 모델별 리포트에서 판정합니다.",
              "세션 수락 오류 ≤5%, 세션 거절 오류 ≤20%는 세션 정보가 없어 N/A입니다.",
              "Image FPR은 other 중 MVP 수락 비율, Image FNR은 MVP 중 other 거절 비율입니다.",
              "폴더 정답 기반 이미지 평가입니다. 비대상 폴더에 실제 대상 객체가 함께 있으면 오류 집계에 영향을 줍니다.",
              "속도는 모델별 기본 입력 크기와 런타임을 사용합니다. 상세 조건은 모델별 metrics.json에 기록됩니다."]
    smoke = any(result.get("smoke_test") for result in results)
    if smoke:
        lines[1:1] = ["", "샘플 실행 검증용 리포트입니다. 전체 데이터 성능을 나타내지 않습니다."]
    if errors:
        lines += ["", "실패한 모델 (비교에서 제외됨):", *[f"- {name}: {message}" for name, message in errors.items()]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Standalone browser report, no remote assets or scripts.
    body = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>COCO MVP report</title>",
            "<style>body{font:16px system-ui;margin:40px;max-width:1200px;color:#172033}.table-wrap{overflow-x:auto}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}th{background:#eef2f7}img{max-width:100%}pre{white-space:pre-wrap}h2{margin-top:48px}a{color:#125ab4}</style>",
            "<h1>COCO MVP 모델 비교</h1><p>computer = laptop + tv · book · other</p>",
            "<p>목표 정확도 ≥90% · 최소 정확도 ≥80%. 세션 오류 기준은 세션 정보가 없어 N/A.</p>",
            "<p>폴더 정답 기반 이미지 분류 지표. 비대상 사진에 실제 대상 객체가 함께 있으면 오류가 과대 집계될 수 있습니다.</p>",
            "<p>Image FPR: other → MVP 수락 비율 · Image FNR: MVP → other 거절 비율. 세션 오류율과 다릅니다.</p>",
            html_table(["Model", "Images", "Accuracy", "Macro precision", "Macro recall", "Macro F1", "Image FPR", "Image FNR", "ms", "FPS", "MiB"],
                       [[row[key] if key in columns[:2] else (percent(row[key]) if key in columns[2:8] else round(row[key], 2))
                         for key in columns] for row in rows])]
    if smoke:
        body.insert(3, "<p><strong>샘플 실행 검증용 결과 — 전체 데이터 성능이 아닙니다.</strong></p>")
    for result in results:
        name = html.escape(result["model"])
        body += [f"<h2>{name}</h2>",
                 html_table(["Class", "Images", "Precision", "Recall", "F1", "FPR", "FNR"],
                     [[r["class"], r["support"], *[percent(r[k]) for k in ("precision", "recall", "f1", "false_accept_rate", "false_reject_rate")]]
                      for r in result["per_class"]]),
                 "<h3>기준 충족 여부</h3>",
                 html_table(["기준", "측정값", "요구값", "판정"],
                     [[c["item"], percent(c["value"]), f"{c['operator']} {percent(c['threshold'])}", c["status"]]
                      for c in result["criteria"]]),
                 f"<img src='{name}/confusion_matrix.png' alt='{name} confusion matrix'>",
                 f"<p><a href='{name}/predictions.csv'>이미지별 예측 CSV</a> · "
                 f"<a href='{name}/per_class.csv'>클래스별 지표 CSV</a> · "
                 f"<a href='{name}/by_source.csv'>출처별 결과 CSV</a> · "
                 f"<a href='{name}/raw_coco_outcomes.csv'>원본 COCO 예측 CSV</a> · "
                 f"<a href='{name}/report.md'>Markdown 리포트</a></p>",
                 "<details><summary>속도 측정 조건 및 런타임</summary><pre>" + html.escape(
                     json.dumps(result["runtime"], ensure_ascii=False, indent=2)) + "</pre></details>"]
    if errors:
        body.append("<h2>실패한 모델</h2><pre>" + html.escape(json.dumps(errors, ensure_ascii=False, indent=2)) + "</pre>")
    body.append("</html>")
    (output / "report.html").write_text("\n".join(body), encoding="utf-8")
