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
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    import torch
    from nanodet.data.batch_process import stack_batch_img
    from nanodet.data.collate import naive_collate
    from nanodet.data.transform import Pipeline
    from nanodet.model.arch import build_model
    from nanodet.util import Logger, cfg, load_config, load_model_weight

    load_config(cfg, str(args.model_config))
    # 전체 COCO weight를 바로 적재하므로 backbone ImageNet weight 재다운로드는 불필요합니다.
    cfg.defrost()
    cfg.model.arch.backbone.pretrain = False
    cfg.freeze()
    logger = Logger(-1, cfg.save_dir, use_tensorboard=False)
    model = build_model(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    load_model_weight(model, checkpoint, logger)
    device = torch.device(args.device)
    model.eval().to(device)
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    rows = []
    with torch.inference_mode():
        for item in read_manifest(args.manifest):
            image = read_cv_image(item["path"])
            height, width = image.shape[:2]
            metadata = {
                "img_info": {
                    "id": 0,
                    "file_name": item["file"],
                    "height": height,
                    "width": width,
                },
                "raw_img": image,
                "img": image,
            }
            started = time.perf_counter()
            metadata = pipeline(None, metadata, cfg.data.val.input_size)
            metadata["img"] = torch.from_numpy(
                metadata["img"].transpose(2, 0, 1)
            ).to(device)
            batch = naive_collate([metadata])
            batch["img"] = stack_batch_img(batch["img"], divisible=32)
            preprocess_ms = (time.perf_counter() - started) * 1000

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = model.inference(batch)[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms = (time.perf_counter() - started) * 1000

            detections = []
            for class_id, boxes in result.items():
                for box in boxes:
                    score = float(box[4])
                    if score >= args.conf:
                        detections.append(
                            (int(class_id), COCO_CLASSES[int(class_id)], score)
                        )
            rows.append(
                topk_row(
                    item, detections, (preprocess_ms, inference_ms, 0.0), args.top_k
                )
            )

    write_adapter_outputs(
        args.output_dir,
        rows,
        model_id=args.model_id,
        framework="NanoDet",
        artifact=args.checkpoint,
        timing_scope="preprocess + NanoDet inference/postprocess; image decode excluded",
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
