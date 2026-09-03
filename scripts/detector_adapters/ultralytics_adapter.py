from __future__ import annotations

import argparse
from pathlib import Path

from common import read_manifest, topk_row, write_adapter_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    from ultralytics import YOLO

    items = read_manifest(args.manifest)
    model = YOLO(str(args.checkpoint))
    rows = []
    for item in items:
        result = model.predict(
            source=item["path"],
            verbose=False,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
        )[0]
        detections = []
        if result.boxes is not None and len(result.boxes):
            ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy().astype(float)
            detections = [
                (int(class_id), str(result.names[int(class_id)]), float(score))
                for class_id, score in zip(ids, scores, strict=True)
            ]
        speed = result.speed
        rows.append(
            topk_row(
                item,
                detections,
                (
                    float(speed.get("preprocess", 0.0)),
                    float(speed.get("inference", 0.0)),
                    float(speed.get("postprocess", 0.0)),
                ),
                args.top_k,
            )
        )
    artifact = Path(getattr(model, "ckpt_path", args.checkpoint))
    write_adapter_outputs(
        args.output_dir,
        rows,
        model_id=args.model_id,
        framework="Ultralytics",
        artifact=artifact,
        timing_scope="preprocess + forward + postprocess; model load and file validation excluded",
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
