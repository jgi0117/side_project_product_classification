from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import COCO_CLASSES, read_manifest, topk_row, write_adapter_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    if str(args.device).casefold() not in {"cpu", "auto"}:
        raise SystemExit("PicoDet ONNX adapter는 현재 CPUExecutionProvider만 지원합니다.")

    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(args.checkpoint),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_names = {item.name for item in session.get_inputs()}
    if not {"image", "scale_factor"}.issubset(input_names):
        raise ValueError(f"예상하지 못한 PicoDet ONNX 입력입니다: {sorted(input_names)}")

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    rows = []
    for item in read_manifest(args.manifest):
        with Image.open(item["path"]) as opened:
            rgb = opened.convert("RGB")
            original_width, original_height = rgb.size

        started = time.perf_counter()
        resized = rgb.resize((320, 320), Image.Resampling.BILINEAR)
        image = np.asarray(resized, dtype=np.float32) / 255.0
        image = ((image - mean) / std).transpose(2, 0, 1)[None, ...]
        scale_factor = np.asarray(
            [[320.0 / original_height, 320.0 / original_width]], dtype=np.float32
        )
        preprocess_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        outputs = session.run(
            None,
            {"image": image.astype(np.float32), "scale_factor": scale_factor},
        )
        inference_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        boxes = np.asarray(outputs[0])
        count = int(np.asarray(outputs[1]).reshape(-1)[0]) if len(outputs) > 1 else len(boxes)
        detections = [
            (int(box[0]), COCO_CLASSES[int(box[0])], float(box[1]))
            for box in boxes[:count]
            if int(box[0]) >= 0 and float(box[1]) >= args.conf
        ]
        postprocess_ms = (time.perf_counter() - started) * 1000
        rows.append(
            topk_row(
                item,
                detections,
                (preprocess_ms, inference_ms, postprocess_ms),
                args.top_k,
            )
        )

    write_adapter_outputs(
        args.output_dir,
        rows,
        model_id=args.model_id,
        framework="ONNX Runtime (PicoDet)",
        artifact=args.checkpoint,
        timing_scope="RGB resize/normalize + ONNX forward/NMS + output parsing; image decode excluded",
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
