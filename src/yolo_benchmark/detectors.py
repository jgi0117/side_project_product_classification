"""RTMDet COCO adapter. Imports stay lazy for its isolated environment."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Contiguous model output indices, not the sparse COCO annotation category IDs.
COCO_NAMES = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
    "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
    "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,"
    "apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
    "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
    "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,"
    "scissors,teddy bear,hair drier,toothbrush"
).split(",")


def detection(label, score, box):
    return {"label": str(label), "score": float(score), "xyxy": [float(v) for v in box]}


def make_detector(spec, config):
    weights = ROOT / spec["weights"]
    if not weights.is_file():
        raise FileNotFoundError(weights)
    device = str(config["device"])
    confidence = float(config["detection_confidence"])
    iou = float(config["nms_iou"])
    size = int(spec["imgsz"])
    if size != 640:
        raise ValueError("RTMDet-tiny's bundled inference pipeline requires imgsz=640")
    kind = spec["adapter"]

    torch_device = "cuda:0" if device == "0" else device
    if kind == "rtmdet":
        import mmdet
        from mmdet.apis import init_detector, inference_detector

        model_config = Path(mmdet.__file__).parent / ".mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py"
        model = init_detector(str(model_config), str(weights), device=torch_device)
        model.test_cfg.score_thr = confidence
        model.test_cfg.nms.iou_threshold = iou

        def predict(image):
            result = inference_detector(model, image).pred_instances.cpu()
            return [detection(COCO_NAMES[int(c)], s, b) for b, s, c in zip(
                result.bboxes.numpy(), result.scores.numpy(), result.labels.numpy())
                if s >= confidence]
        return predict

    raise ValueError(f"Unsupported adapter: {kind}; this branch supports RTMDet only")
