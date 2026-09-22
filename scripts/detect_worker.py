"""Run one detector in its own Python environment; exchange JSON, not imports."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# YAPF/MMEngine cache creation must stay inside the writable workspace.
cache_dir = ROOT / "outputs" / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WIN_PD_OVERRIDE_LOCAL_APPDATA", str(cache_dir))
os.environ["TEMP"] = str(cache_dir)
os.environ["TMP"] = str(cache_dir)
sys.path.insert(0, str(ROOT / "src"))
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

import cv2
import numpy as np
from tqdm import tqdm

from yolo_benchmark.common import seed_everything, write_json
from yolo_benchmark.detectors import make_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    config, spec = job["config"], job["model"]
    if str(config["device"]).isdigit():
        config["device"] = "cuda:" + str(config["device"])
    print(f"Initializing {spec['name']} in {sys.executable}", flush=True)
    seed_everything(int(config["seed"]))
    print("Loading detector", flush=True)
    predict = make_detector(spec, config)
    print("Detector ready", flush=True)
    faulthandler.cancel_dump_traceback_later()
    sync = lambda: None
    if str(config["device"]) != "cpu":
        import torch
        sync = lambda: torch.cuda.synchronize(config["device"])
    warmup = int(config["speed_warmup"])
    repeats = int(config["speed_repeats"])
    durations = []
    images = []
    for index, item in enumerate(tqdm(job["images"], desc=spec["name"], unit="image")):
        # imdecode supports Korean and other Unicode Windows paths.
        image = cv2.imdecode(np.fromfile(item["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode validated image: {item['path']}")
        if index == 0:
            for _ in range(warmup):
                predict(image)
            sync()
        for repeat in range(repeats):
            sync()
            start = time.perf_counter()
            detections = predict(image)
            sync()
            durations.append((time.perf_counter() - start) * 1000)
            if repeat == 0:
                images.append({**item, "detections": detections})
    mean = float(np.mean(durations))
    runtime = {"latency_mean_ms": mean, "latency_p50_ms": float(np.percentile(durations, 50)),
               "latency_p95_ms": float(np.percentile(durations, 95)), "fps": 1000 / mean,
               "model_size_mib": (ROOT / spec["weights"]).stat().st_size / 1024**2,
               "adapter": spec["adapter"], "imgsz": spec["imgsz"], "device": config["device"],
               "python": platform.python_version(), "warmup": warmup, "repeats": repeats,
               "timed_calls": len(durations), "scope": "decoded image preprocessing + forward + postprocessing",
               "postprocessing": "RTMDet configured score threshold and NMS"}
    runtime["effective_score_floor"] = float(config["detection_confidence"])
    runtime["weights_sha256"] = hashlib.sha256((ROOT / spec["weights"]).read_bytes()).hexdigest()
    runtime["packages"] = {name: importlib.metadata.version(name)
                           for name in ("torch", "torchvision", "mmcv", "mmengine", "mmdet")}
    write_json(Path(job["result"]), {"model": spec["name"], "runtime": runtime, "images": images})
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
