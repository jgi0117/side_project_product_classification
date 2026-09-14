"""COCO detector adapters. Imports stay lazy for isolated model environments."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
    kind = spec["adapter"]

    if kind == "ultralytics":
        from ultralytics import YOLO

        model = YOLO(str(weights), task="detect")
        if model.task != "detect" or list(model.names.values()) != COCO_NAMES:
            raise ValueError("Expected an 80-class COCO detection checkpoint")

        def predict(image):
            result = model.predict(image, device=device, imgsz=size, conf=confidence,
                                   iou=iou, verbose=False, save=False)[0]
            return [detection(result.names[int(c)], s, b) for b, s, c in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy())]
        return predict

    if kind == "picodet":
        import cv2
        import onnxruntime as ort

        if device != "cpu":
            raise ValueError("This PicoDet adapter uses CPUExecutionProvider; use --device cpu")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(str(weights), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        if {x.name for x in session.get_inputs()} != {"image", "scale_factor"}:
            raise ValueError("Expected the postprocessed PicoDet COCO ONNX export")

        def predict(image):
            height, width = image.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
            normalized = (rgb.astype(np.float32) / 255 - np.array(
                [0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
            boxes, count = session.run(None, {
                "image": normalized.transpose(2, 0, 1)[None].copy(),
                "scale_factor": np.array([[size / height, size / width]], np.float32),
            })
            return [detection(COCO_NAMES[int(row[0])], row[1], row[2:6])
                    for row in boxes[:int(count.reshape(-1)[0])]
                    if 0 <= int(row[0]) < 80 and row[1] >= confidence]
        return predict

    import torch
    torch_device = "cuda:0" if device == "0" else device
    if kind == "yolox":
        sys.path.insert(0, str(ROOT / "third_party/YOLOX"))
        from yolox.exp import get_exp
        from yolox.data.data_augment import ValTransform
        from yolox.utils import postprocess

        exp = get_exp(None, spec["name"])
        model = exp.get_model().to(torch_device).eval()
        model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=False)["model"])
        transform = ValTransform(legacy=False)

        def predict(image):
            ratio = min(size / image.shape[0], size / image.shape[1])
            array, _ = transform(image, None, (size, size))
            tensor = torch.from_numpy(array).unsqueeze(0).float().to(torch_device)
            with torch.no_grad():
                output = postprocess(model(tensor), 80, confidence, iou,
                                     class_agnostic=False)[0]
            if output is None:
                return []
            return [detection(COCO_NAMES[int(row[6])], row[4] * row[5], row[:4] / ratio)
                    for row in output.cpu().numpy()]
        return predict

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

    if kind == "nanodet":
        sys.path.insert(0, str(ROOT / "third_party/nanodet"))
        from nanodet.util import cfg, load_config, load_model_weight
        from nanodet.model.arch import build_model
        from nanodet.data.transform import Pipeline
        from nanodet.data.collate import naive_collate
        from nanodet.data.batch_process import stack_batch_img

        load_config(cfg, str(ROOT / "third_party/nanodet/config/nanodet-plus-m_320.yml"))
        # Avoid downloading backbone initialization: the full detector is loaded below.
        cfg.defrost()
        cfg.model.arch.backbone.pretrain = False
        model = build_model(cfg.model)

        class Logger:
            def log(self, message):
                print(message)

        load_model_weight(model, torch.load(str(weights), map_location="cpu", weights_only=False), Logger())
        model = model.to(torch_device).eval()
        pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

        def predict(image):
            height, width = image.shape[:2]
            meta = {"img_info": {"id": 0, "height": height, "width": width},
                    "raw_img": image, "img": image}
            meta = pipeline(None, meta, cfg.data.val.input_size)
            meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(torch_device)
            meta = naive_collate([meta])
            meta["img"] = stack_batch_img(meta["img"], divisible=32)
            with torch.no_grad():
                output = model.inference(meta)[0]
            return [detection(COCO_NAMES[int(label)], box[4], box[:4])
                    for label, boxes in output.items() for box in boxes if box[4] >= confidence]
        return predict
    raise ValueError(f"Unsupported adapter: {kind}")
