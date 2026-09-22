"""Apply a trained image-level postprocessor to RTMDet detections."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.common import write_json  # noqa: E402
from yolo_benchmark.postprocessor import extract_features, ranked_candidates  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    bundle = joblib.load(args.model)
    raw = json.loads(args.detections.read_text(encoding="utf-8"))
    expected_weights = bundle["training"].get("rtmdet_weights_sha256")
    actual_weights = raw.get("runtime", {}).get("weights_sha256")
    if expected_weights and actual_weights != expected_weights:
        raise ValueError("RTMDet weights do not match the trained postprocessor")
    expected_floor = bundle["training"].get("effective_score_floor")
    actual_floor = raw.get("runtime", {}).get("effective_score_floor")
    if expected_floor is not None and actual_floor != expected_floor:
        raise ValueError("Detection score floor does not match training")
    features, names = extract_features(raw["images"])
    if names != bundle["feature_names"]:
        raise ValueError("Feature schema does not match trained model")
    model = bundle["model"]
    probabilities = model.predict_proba(features)
    indices = [list(model.classes_).index(label) for label in bundle["classes"]]
    probabilities = probabilities[:, indices]
    candidates = ranked_candidates(probabilities, bundle["classes"], args.top_k)
    write_json(args.output, {
        "model": str(args.model), "top_k": args.top_k, "classes": bundle["classes"],
        "images": [{"sha256": image.get("sha256"), "candidates": candidates[index],
                    "probabilities": {label: float(probabilities[index, class_index])
                                      for class_index, label in enumerate(bundle["classes"])}}
                   for index, image in enumerate(raw["images"])]})
    print(args.output)


if __name__ == "__main__":
    main()
