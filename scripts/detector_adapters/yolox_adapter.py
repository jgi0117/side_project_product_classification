from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import (
    COCO_CLASSES,
    read_cv_image,
    read_manifest,
    topk_row,
    write_adapter_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    import torch
    from yolox.data.data_augment import ValTransform
    from yolox.exp import get_exp
    from yolox.utils import postprocess

    device = torch.device(args.device)
    exp = get_exp(None, args.exp_name)
    exp.test_conf = args.conf
    exp.nmsthre = args.iou
    model = exp.get_model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    transform = ValTransform(legacy=False)

    rows = []
    with torch.inference_mode():
        for item in read_manifest(args.manifest):
            image = read_cv_image(item["path"])
            started = time.perf_counter()
            tensor, _ = transform(image, None, exp.test_size)
            preprocess_ms = (time.perf_counter() - started) * 1000

            tensor = torch.from_numpy(tensor).unsqueeze(0).float().to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            output = postprocess(
                output,
                exp.num_classes,
                exp.test_conf,
                exp.nmsthre,
                class_agnostic=True,
            )[0]
            detections = []
            if output is not None:
                output = output.detach().cpu()
                class_ids = output[:, 6].int().tolist()
                scores = (output[:, 4] * output[:, 5]).tolist()
                detections = [
                    (class_id, COCO_CLASSES[class_id], float(score))
                    for class_id, score in zip(class_ids, scores, strict=True)
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
        framework="YOLOX",
        artifact=args.checkpoint,
        timing_scope="preprocess + forward + NMS; image decode excluded",
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
