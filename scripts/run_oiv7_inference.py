from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.benchmark import resolve_device  # noqa: E402
from yolo_benchmark.common import load_config, seed_everything  # noqa: E402
from yolo_benchmark.data import discover_images  # noqa: E402
from yolo_benchmark.oiv7 import evaluate_oiv7  # noqa: E402


def main(
    default_config: Path = ROOT / "config" / "oiv7.yaml",
    description: str = "OIV7 사전학습 YOLOv8n의 전체 601-class top-1 인증 평가",
) -> None:
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--device")
    args = parser.parse_args()

    config = load_config(args.config)
    if int(config["top_k"]) != 1:
        raise SystemExit("이 실험은 top_k: 1만 지원합니다.")
    source = (args.source or config["raw_dir"]).resolve()
    if not source.is_dir():
        raise SystemExit(f"이미지 루트 폴더를 찾지 못했습니다: {source}")

    seed_everything(int(config["seed"]))
    device = resolve_device(args.device or str(config["device"]))
    items, invalid_images = discover_images(source, list(config["classes"]))

    # 지연 import로 --help나 설정 검증만 할 때 모델 패키지를 강제하지 않습니다.
    from ultralytics import YOLO

    model_name = args.model or str(config["model"])
    model = YOLO(model_name)
    metrics = evaluate_oiv7(
        model=model,
        source=source,
        items=items,
        classes=list(config["classes"]),
        target_model_labels=dict(config["target_model_labels"]),
        thresholds=dict(config["verification_thresholds"]),
        output_dir=config["output_dir"] / "pretrained",
        device=device,
        imgsz=int(config["imgsz"]),
        confidence_floor=float(config["model_confidence_floor"]),
        iou=float(config["iou"]),
        max_det=int(config["max_det"]),
    )
    print(
        f"valid={len(items)}, invalid={invalid_images}, "
        f"macro_accept={metrics['expected_label_accept_rate_macro']:.3f}"
    )
    print(f"결과: {(config['output_dir'] / 'pretrained').resolve()}")


if __name__ == "__main__":
    main()
