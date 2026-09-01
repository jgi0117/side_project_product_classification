from pathlib import Path

from run_oiv7_inference import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(
        default_config=root / "config" / "coco.yaml",
        description=(
            "COCO 사전학습 YOLOv8n의 전체 80-class top-1 평가 "
            "(guitar 제외)"
        ),
    )
