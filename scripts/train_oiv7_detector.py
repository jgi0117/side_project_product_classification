from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def validate_oiv7_dataset(data: str) -> str:
    """Reject local dataset YAMLs that would replace the 601-class head."""
    if data == "open-images-v7.yaml":
        return data

    path = Path(data).resolve()
    if not path.is_file():
        raise SystemExit(f"dataset YAML을 찾지 못했습니다: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    names = config.get("names")
    if isinstance(names, dict):
        labels = list(names.values())
    elif isinstance(names, list):
        labels = names
    else:
        labels = []
    normalized = {str(label).strip().casefold() for label in labels}
    required = {"bicycle", "book", "guitar", "laptop"}
    if len(labels) != 601 or not required.issubset(normalized):
        raise SystemExit(
            "601-class OIV7 head를 유지하려면 names가 OIV7 전체 601개 클래스인 "
            "dataset YAML이 필요합니다. 4-class 폴더 데이터로는 학습할 수 없습니다."
        )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OIV7 사전학습 YOLOv8n의 601-class detection head 학습"
    )
    parser.add_argument(
        "--data",
        default="open-images-v7.yaml",
        help="기본값은 Ultralytics 내장 Open Images V7 dataset YAML",
    )
    parser.add_argument("--model", default="yolov8n-oiv7.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", type=Path, default=Path("outputs/oiv7/train"))
    parser.add_argument("--name", default="yolov8n-oiv7-601-class")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    data = validate_oiv7_dataset(args.data)
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise SystemExit("epochs, imgsz, batch 값이 올바르지 않습니다.")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        seed=42,
        deterministic=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
