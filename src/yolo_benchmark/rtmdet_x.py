"""RTMDet-x COCO detections mapped to the sprint's single MVP label."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models/pretrained/rtmdet_x.pth"
CONFIG_NAME = "rtmdet_x_8xb32-300e_coco.py"
MODEL_NAME = "rtmdet-x"
WEIGHTS_SHA256 = "cc79b9ae0970f7f597e2aa5f81663129f3aa24e5612549b4f1a383da7fff825d"
DETECTION_CONFIDENCE = 0.01
NMS_IOU = 0.6
GROUP_THRESHOLD = 0.05
GROUPS = ("computer", "book", "other")

# Contiguous COCO model output indices, not sparse COCO annotation IDs.
GROUP_BY_CLASS_ID = {62: "computer", 63: "computer", 73: "book"}


def decide_top1(detections: list[tuple[int, float]]) -> dict:
    """Use the highest confidence per group, as in the RTMDet size experiment."""
    scores = dict.fromkeys(GROUPS, 0.0)
    for class_id, confidence in detections:
        if not 0 <= class_id < 80:
            raise ValueError(f"Unexpected COCO class index: {class_id}")
        if confidence < DETECTION_CONFIDENCE:
            continue
        group = GROUP_BY_CLASS_ID.get(class_id, "other")
        scores[group] = max(scores[group], float(confidence))
    passing = [group for group in GROUPS if scores[group] >= GROUP_THRESHOLD]
    prediction = max(passing, key=scores.__getitem__) if passing else "other"
    return {
        "model": MODEL_NAME,
        "top_k": 1,
        "prediction": prediction,
        "confidence": scores[prediction],
        "group_scores": scores,
        "decision_reason": "highest_passing_group" if passing else "no_passing_detection",
    }


class RTMDetXPredictor:
    def __init__(self, weights: Path | None = None, device: str | None = None):
        import mmdet
        from mmdet.apis import inference_detector, init_detector

        configured = weights or Path(os.getenv("RTMDET_X_WEIGHTS", str(DEFAULT_WEIGHTS))).expanduser()
        weights_path = configured if configured.is_absolute() else PROJECT_ROOT / configured
        if not weights_path.is_file():
            raise FileNotFoundError(f"RTMDet-x checkpoint missing: {weights_path}")
        with weights_path.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        if checksum != WEIGHTS_SHA256:
            raise ValueError(f"Unexpected RTMDet-x checkpoint SHA-256: {checksum}")
        config_path = Path(mmdet.__file__).parent / ".mim/configs/rtmdet" / CONFIG_NAME
        if not config_path.is_file():
            raise FileNotFoundError(f"MMDetection RTMDet-x config missing: {config_path}")
        self.model = init_detector(str(config_path), str(weights_path), device=device or os.getenv("RTMDET_DEVICE", "cpu"))
        self.model.test_cfg.score_thr = DETECTION_CONFIDENCE
        self.model.test_cfg.nms.iou_threshold = NMS_IOU
        self._inference_detector = inference_detector
        self._lock = Lock()

    def predict(self, image) -> dict:
        # MMDetection's model is shared by requests; serialize access to it.
        with self._lock:
            instances = self._inference_detector(self.model, image).pred_instances.cpu()
        detections = [(int(class_id), float(score)) for class_id, score in zip(
            instances.labels.tolist(), instances.scores.tolist())]
        return decide_top1(detections)
