from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.benchmark import (  # noqa: E402
    evaluate_zero_shot,
    resolve_device,
    save_summary,
)
from yolo_benchmark.common import DEFAULT_CONFIG, load_config, seed_everything  # noqa: E402
from yolo_benchmark.data import discover_images  # noqa: E402


def find_mounted_sample(configured: Path) -> Path | None:
    """Find a Drive for desktop mount when its letter or root name differs."""
    candidates = [configured]
    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:/")
        candidates.extend(
            [
                drive / "My Drive" / "side_project" / "sample",
                drive / "내 드라이브" / "side_project" / "sample",
                drive / "My Drive" / "side_project",
                drive / "내 드라이브" / "side_project",
                drive / "side_project" / "sample",
                drive / "side_project",
            ]
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8n/11n/26n ImageNet zero-shot inference 비교"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source", type=Path, help="mount된 원본 카테고리 폴더")
    parser.add_argument("--models", nargs="+", help="비교할 *-cls.pt 체크포인트")
    parser.add_argument("--device", help="auto, cpu, 0 등")
    parser.add_argument(
        "--backend",
        choices=("pytorch", "onnx", "both"),
        help="실행 backend; 생략 시 inference.yaml의 backends 사용",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    configured_source = (args.source or config["raw_dir"]).resolve()
    source = find_mounted_sample(configured_source)
    if source is None:
        raise SystemExit(
            "Google Drive의 mount된 sample 폴더를 찾지 못했습니다.\n"
            f"설정 경로: {configured_source}\n"
            f"브라우저 링크: {config.get('drive_folder_url', '(없음)')}\n"
            "Google Drive for desktop을 실행한 뒤 PowerShell에서 "
            "`Get-PSDrive -PSProvider FileSystem`과 `Get-ChildItem G:\\`로 "
            "실제 드라이브 문자와 내 드라이브 폴더명을 확인하세요."
        )
    if source != configured_source:
        print(f"Configured path not found; auto-detected Drive path: {source}")

    models = args.models or config["models"]
    invalid = [name for name in models if not Path(name).stem.endswith("-cls")]
    if invalid:
        raise SystemExit(f"분류 체크포인트(*-cls.pt)만 사용할 수 있습니다: {invalid}")
    if args.backend == "both":
        backends = ["pytorch", "onnx"]
    elif args.backend:
        backends = [args.backend]
    else:
        backends = list(config["backends"])
    unsupported = [name for name in backends if name not in {"pytorch", "onnx"}]
    if unsupported:
        raise SystemExit(f"지원하지 않는 backend입니다: {unsupported}")

    seed_everything(int(config["seed"]))
    device = resolve_device(args.device or str(config["device"]))
    print(
        f"Source: {source}\nDevice: {device}\nBackends: {backends}\n"
        "Mode: ImageNet zero-shot open-set verification (no training)"
    )
    items, invalid_images = discover_images(source, config["classes"])
    print(f"Unique valid images: {len(items)}, invalid skipped: {invalid_images}")
    results_by_backend = {backend: [] for backend in backends}
    for backend in backends:
        for model_name in models:
            print(f"\n===== {model_name} [{backend}] =====")
            results_by_backend[backend].append(
                evaluate_zero_shot(
                    model_name=model_name,
                    backend=backend,
                    source=source,
                    model_dir=config["model_dir"],
                    output_dir=config["output_dir"],
                    classes=config["classes"],
                    patterns=config["imagenet_class_patterns"],
                    thresholds=config["verification_thresholds"],
                    top_k=int(config["top_k"]),
                    onnx_simplify=bool(config["onnx"]["simplify"]),
                    device=device,
                    imgsz=int(config["imgsz"]),
                    speed_warmup=int(config["speed_warmup"]),
                    speed_repeats=int(config["speed_repeats"]),
                    items=items,
                    invalid_images=invalid_images,
                )
            )
            save_summary(
                results_by_backend[backend], config["output_dir"] / backend
            )
    print("\nSummaries:")
    for backend in backends:
        summary = (config["output_dir"] / backend / "summary.csv").resolve()
        print(f"- {backend}: {summary}")


if __name__ == "__main__":
    main()
