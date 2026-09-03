from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.common import write_json  # noqa: E402
from yolo_benchmark.data import discover_images  # noqa: E402
from yolo_benchmark.downloads import ensure_download  # noqa: E402
from yolo_benchmark.oiv7 import save_detector_visualizations  # noqa: E402


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_manifest(path: Path, source: Path, items: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "file", "true_label"])
        writer.writeheader()
        writer.writerows(
            {
                "path": str(item.source),
                "file": item.source.relative_to(source).as_posix(),
                "true_label": item.label,
            }
            for item in items
        )


def adapter_command(
    model: dict[str, Any], manifest: Path, output_dir: Path, device: str, config: dict[str, Any]
) -> list[str]:
    python = resolve_path(model["python"])
    adapter = ROOT / "scripts" / "detector_adapters" / f"{model['adapter']}_adapter.py"
    command = [
        str(python),
        str(adapter),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--model-id",
        model["id"],
        "--checkpoint",
        str(resolve_path(model["checkpoint"])),
        "--device",
        device,
        "--conf",
        str(config["model_confidence_floor"]),
        "--top-k",
        str(config["top_k"]),
    ]
    for key in ("model_config", "exp_name", "paddledetection_root"):
        if key in model:
            flag = "--" + key.replace("_", "-")
            value = model[key]
            if key in {"paddledetection_root"} or (
                key == "model_config" and str(value).endswith((".yml", ".yaml", ".py"))
            ):
                value = resolve_path(value)
            command.extend([flag, str(value)])
    if model["adapter"] in {"ultralytics"}:
        command.extend(
            [
                "--imgsz",
                str(config["imgsz"]),
                "--iou",
                str(config["iou"]),
                "--max-det",
                str(config["max_det"]),
            ]
        )
    elif model["adapter"] == "yolox":
        command.extend(["--iou", str(config["iou"])])
    return command


def missing_model_paths(model: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    python = resolve_path(model["python"])
    if not python.is_file():
        missing.append(f"Python 실행 파일 없음: {python}")

    adapter = ROOT / "scripts" / "detector_adapters" / f"{model['adapter']}_adapter.py"
    if not adapter.is_file():
        missing.append(f"adapter 없음: {adapter}")

    checkpoint = resolve_path(model["checkpoint"])
    if not checkpoint.exists() and not model.get("allow_checkpoint_download", False):
        missing.append(f"모델 가중치 없음: {checkpoint}")

    model_config = model.get("model_config")
    if model_config and str(model_config).endswith((".yml", ".yaml", ".py")):
        config_path = resolve_path(model_config)
        if not config_path.is_file():
            missing.append(f"모델 설정 파일 없음: {config_path}")

    paddle_root = model.get("paddledetection_root")
    if paddle_root:
        deploy_script = resolve_path(paddle_root) / "deploy" / "python" / "infer.py"
        if not deploy_script.is_file():
            missing.append(f"PaddleDetection 저장소 없음: {resolve_path(paddle_root)}")
    return missing


def download_model_artifacts(models: list[dict[str, Any]]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for model in models:
        spec = model.get("download")
        if not spec:
            continue
        try:
            path, downloaded = ensure_download(spec, ROOT)
            state = "다운로드 완료" if downloaded else "기존 파일 확인"
            print(f"[{model['id']}] {state}: {path}")
        except Exception as error:  # 각 모델의 실패가 다음 다운로드를 막지 않게 합니다.
            errors[model["id"]] = str(error)
            print(f"[{model['id']}] 다운로드 실패: {error}", file=sys.stderr)
    return errors


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finalize_model_report(
    output_dir: Path,
    classes: list[str],
    target_labels: dict[str, str],
    thresholds: dict[str, float],
    top_k: int,
) -> dict[str, Any]:
    rows = read_csv(output_dir / "raw_predictions.csv")
    numeric_fields = {
        "all_detection_count": int,
        "unique_detection_class_count": int,
        "preprocess_ms": float,
        "inference_ms": float,
        "postprocess_ms": float,
        "total_pipeline_ms": float,
    }
    for rank in range(1, top_k + 1):
        numeric_fields[f"model_top{rank}_class_id"] = int
        numeric_fields[f"model_top{rank}_score"] = float
    for row in rows:
        for field, converter in numeric_fields.items():
            row[field] = converter(row[field])
        expected = target_labels[row["true_label"]].casefold()
        passed = any(
            row[f"model_top{rank}_label"].strip().casefold() == expected
            and row[f"model_top{rank}_score"] >= float(thresholds[row["true_label"]])
            for rank in range(1, top_k + 1)
        )
        row["expected_label_pass"] = passed
        row["decision"] = "pass" if passed else "reject"

    fields = list(rows[0])
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    class_summary = save_detector_visualizations(rows, classes, target_labels, output_dir)
    with (output_dir / "adapter_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    latency = np.asarray([row["total_pipeline_ms"] for row in rows], dtype=float)
    counts = Counter(row["true_label"] for row in rows)
    accept_rates = {
        label: sum(
            row["true_label"] == label and row["expected_label_pass"] for row in rows
        )
        / counts[label]
        for label in classes
    }
    metrics = {
        **metadata,
        "top_k": top_k,
        "thresholds": thresholds,
        "expected_label_accept_rate_by_class": accept_rates,
        "expected_label_accept_rate_macro": float(np.mean(list(accept_rates.values()))),
        "service_reject_rate": float(np.mean([not row["expected_label_pass"] for row in rows])),
        "no_detection_rate": float(np.mean([row["all_detection_count"] == 0 for row in rows])),
        "latency_mean_ms": float(np.mean(latency)),
        "latency_p50_ms": float(np.percentile(latency, 50)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "fps_from_mean_latency": 1000.0 / float(np.mean(latency)),
        "model_top1_label_counts": dict(Counter(row["model_top1_label"] for row in rows).most_common()),
        "model_topk_label_counts": dict(
            Counter(
                row[f"model_top{rank}_label"]
                for row in rows
                for rank in range(1, top_k + 1)
                if row[f"model_top{rank}_class_id"] >= 0
            ).most_common()
        ),
        "class_summary": class_summary,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def write_comparison(output_dir: Path, metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        return
    classes = list(metrics[0]["expected_label_accept_rate_by_class"])
    rows = []
    for item in metrics:
        row = {
            "model": item["model_id"],
            "framework": item["framework"],
            "top_k": item["top_k"],
            "images": item["images"],
            "macro_pass_rate": item["expected_label_accept_rate_macro"],
            "latency_mean_ms": item["latency_mean_ms"],
            "latency_p50_ms": item["latency_p50_ms"],
            "latency_p95_ms": item["latency_p95_ms"],
            "model_size_mib": item["model_size_mib"],
        }
        row.update(
            {
                f"{label}_pass_rate": item["expected_label_accept_rate_by_class"][label]
                for label in classes
            }
        )
        rows.append(row)
    with (output_dir / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "model_comparison.json", rows)

    names = [row["model"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(names, [row["macro_pass_rate"] for row in rows], color="#59A14F")
    axes[0, 0].set_title("Macro Pass Rate")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].bar(names, [row["latency_mean_ms"] for row in rows], color="#4E79A7")
    axes[0, 1].set_title("Mean Pipeline Time per Image")
    axes[0, 1].set_ylabel("ms")
    axes[1, 0].bar(names, [row["model_size_mib"] for row in rows], color="#F28E2B")
    axes[1, 0].set_title("Model Artifact Size")
    axes[1, 0].set_ylabel("MiB")
    matrix = np.asarray(
        [[row[f"{label}_pass_rate"] for label in classes] for row in rows]
    )
    image = axes[1, 1].imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes[1, 1].set_xticks(range(len(classes)), classes)
    axes[1, 1].set_yticks(range(len(names)), names)
    axes[1, 1].set_title("Pass Rate by Folder Label")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axes[1, 1].text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.0%}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO 경량 detector 순차 zero-shot 평가")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "coco_models.yaml")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--models", nargs="+", help="모델 id 일부만 실행")
    parser.add_argument("--top-k", type=int, help="이미지별 서로 다른 클래스 예측 개수")
    parser.add_argument("--output-dir", type=Path, help="결과 저장 경로 직접 지정")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="추론하지 않고 Python·가중치·외부 저장소 경로만 검사",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="누락된 공식 가중치만 내려받고 추론하지 않음",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="누락된 가중치 자동 다운로드를 사용하지 않음",
    )
    args = parser.parse_args()

    with args.config.resolve().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    top_k = args.top_k if args.top_k is not None else int(config.get("top_k", 1))
    if top_k < 1:
        raise SystemExit("--top-k는 1 이상이어야 합니다.")
    config["top_k"] = top_k
    base_output_dir = resolve_path(config["output_dir"])
    if args.output_dir:
        output_dir = resolve_path(args.output_dir)
    elif top_k == 1:
        output_dir = base_output_dir
    else:
        output_dir = base_output_dir.with_name(f"{base_output_dir.name}_top{top_k}")
    classes = list(config["classes"])
    selected = [
        model for model in config["models"] if not args.models or model["id"] in args.models
    ]
    unknown = set(args.models or []) - {model["id"] for model in selected}
    if unknown:
        raise SystemExit(f"설정에 없는 모델입니다: {sorted(unknown)}")

    download_errors: dict[str, str] = {}
    if not args.check_only and not args.no_download:
        download_errors = download_model_artifacts(selected)
    if args.download_only:
        if download_errors:
            raise SystemExit("일부 모델 가중치 다운로드에 실패했습니다.")
        return

    path_errors = {model["id"]: missing_model_paths(model) for model in selected}
    for model_id, error in download_errors.items():
        path_errors[model_id].insert(0, f"자동 다운로드 실패: {error}")
    missing_runtime = any(
        any(
            error.startswith(("Python 실행 파일 없음:", "모델 설정 파일 없음:"))
            for error in errors
        )
        for errors in path_errors.values()
    )
    if missing_runtime:
        print(
            "모델별 실행 환경이 없습니다. 먼저 다음 명령을 실행하세요:\n"
            "  python scripts\\setup_coco_model_envs.py",
            file=sys.stderr,
        )
    if args.check_only:
        for model in selected:
            errors = path_errors[model["id"]]
            if errors:
                print(f"[MISSING] {model['id']}")
                for error in errors:
                    print(f"  - {error}")
            else:
                print(f"[READY]   {model['id']}")
        if any(path_errors.values()):
            raise SystemExit("누락된 실행 환경 또는 모델 파일이 있습니다.")
        return

    source = (args.source or resolve_path(config["raw_dir"])).resolve()
    items, invalid = discover_images(source, classes)
    manifest = output_dir / "manifest.csv"
    write_manifest(manifest, source, items)

    statuses = []
    successful_metrics = []
    for index, model in enumerate(selected, start=1):
        model_dir = output_dir / model["id"]
        print(f"[{index}/{len(selected)}] {model['id']} 추론 시작")
        if path_errors[model["id"]]:
            message = " | ".join(path_errors[model["id"]])
            statuses.append({"model": model["id"], "status": "failed", "message": message})
            print(f"[{model['id']}] 실행 전 경로 검사 실패: {message}", file=sys.stderr)
            continue
        command = adapter_command(
            model,
            manifest,
            model_dir,
            args.device or config["device"],
            config,
        )
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            metrics = finalize_model_report(
                model_dir,
                classes,
                dict(config["target_model_labels"]),
                dict(config["verification_thresholds"]),
                top_k,
            )
            successful_metrics.append(metrics)
            statuses.append({"model": model["id"], "status": "success", "message": ""})
        except (OSError, subprocess.CalledProcessError, FileNotFoundError) as error:
            statuses.append({"model": model["id"], "status": "failed", "message": str(error)})
            print(f"[{model['id']}] 실패: {error}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_status.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "status", "message"])
        writer.writeheader()
        writer.writerows(statuses)
    write_comparison(output_dir, successful_metrics)
    print(f"valid={len(items)}, invalid={invalid}, success={len(successful_metrics)}/{len(selected)}")
    print(f"결과: {output_dir.resolve()}")
    if len(successful_metrics) != len(selected):
        raise SystemExit("일부 모델이 실패했습니다. run_status.csv와 위 오류를 확인하세요.")


if __name__ == "__main__":
    main()
