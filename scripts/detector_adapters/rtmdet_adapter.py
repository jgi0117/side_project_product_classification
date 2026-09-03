from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

from common import read_cv_image, read_manifest, topk_row, write_adapter_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    temporary_dir = args.output_dir / ".tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temporary_dir.resolve())
    os.environ["TMP"] = str(temporary_dir.resolve())
    tempfile.tempdir = str(temporary_dir.resolve())

    from mmdet.apis import DetInferencer

    items = read_manifest(args.manifest)
    inferencer = DetInferencer(
        model=args.model_config,
        weights=str(args.checkpoint),
        device=args.device,
    )
    names = inferencer.model.dataset_meta["classes"]
    rows = []
    for item in items:
        image = read_cv_image(item["path"])
        started = time.perf_counter()
        output = inferencer(
            image,
            pred_score_thr=args.conf,
            no_save_pred=True,
            no_save_vis=True,
            return_vis=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        prediction = output["predictions"][0]
        detections = [
            (int(class_id), str(names[int(class_id)]), float(score))
            for class_id, score in zip(
                prediction["labels"], prediction["scores"], strict=True
            )
        ]
        rows.append(topk_row(item, detections, (0.0, elapsed_ms, 0.0), args.top_k))

    write_adapter_outputs(
        args.output_dir,
        rows,
        model_id=args.model_id,
        framework="MMDetection",
        artifact=args.checkpoint,
        timing_scope="DetInferencer preprocess + forward + postprocess; image decode excluded",
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
