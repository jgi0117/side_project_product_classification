from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.common import DEFAULT_CONFIG, load_config, write_json
from yolo_benchmark.data import discover_images
from yolo_benchmark.reports import render_model, render_summary


def find_source(configured):
    if configured.is_dir():
        return configured
    # Locate translated Drive root names without assuming its mount letter.
    for letter in "GHIJKLMNOPQRSTUVWXYZABCDEF":
        drive = Path(f"{letter}:/")
        for name in ("My Drive", "내 드라이브"):
            candidate = drive / name / "side_project/sample"
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"Drive sample 폴더를 찾을 수 없습니다. --source로 지정하세요: {configured}")


def validate_config(config):
    if config["classes"] != ["computer", "book"]:
        raise ValueError("MVP classes must be [computer, book]")
    if config["coco_class_mapping"] != {"computer": ["laptop", "tv"], "book": ["book"]}:
        raise ValueError("Expected COCO mapping computer=laptop+tv, book=book")
    for value in [*config["verification_thresholds"].values(), config["other_threshold"],
                  config["detection_confidence"], config["nms_iou"], *config["report_criteria"].values()]:
        if not 0 < float(value) <= 1:
            raise ValueError("Thresholds must be in (0, 1]")
    if config["detection_confidence"] > min(*config["verification_thresholds"].values(), config["other_threshold"]):
        raise ValueError("Detection confidence must not exceed decision thresholds")
    if config["speed_repeats"] < 1 or config["speed_warmup"] < 0:
        raise ValueError("speed_repeats >= 1 and speed_warmup >= 0 required")
    names = [spec["name"] for spec in config["models"]]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in names):
        raise ValueError("Model names must be unique safe folder names")


def main():
    parser = argparse.ArgumentParser(description="COCO MVP inference and precision/recall reports")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--models", nargs="+", help="Model names from config")
    parser.add_argument("--device", help="cpu, cuda:0, ...")
    parser.add_argument("--output-name", help="New run folder under output_dir")
    parser.add_argument("--limit-per-category", type=int, help="Smoke test only; sample each source category")
    parser.add_argument("--report-only", type=Path, help="Regenerate reports from an existing run's raw detections")
    args = parser.parse_args()
    if args.report_only:
        output = args.report_only.resolve()
        snapshot = json.loads((output / "run.json").read_text(encoding="utf-8"))
        config = snapshot["config"]
        results, errors = [], {}
        for model in snapshot["selected_models"]:
            raw_path = output / model / "detections.json"
            if not raw_path.is_file():
                errors[model] = "No completed detections.json; see inference.log"
                continue
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            results.append(render_model(raw, config, output / model))
        render_summary(results, errors, output)
        print(f"Report: {output / 'report.html'}")
        return 1 if errors else 0

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    validate_config(config)
    selected = [spec for spec in config["models"] if not args.models or spec["name"] in args.models]
    if args.models and set(args.models) - {spec["name"] for spec in selected}:
        parser.error("Unknown model name; check config/inference.yaml")
    if args.limit_per_category is not None and args.limit_per_category < 1:
        parser.error("--limit-per-category must be positive")
    source = args.source.resolve() if args.source else find_source(config["raw_dir"])
    items, invalid = discover_images(source, config["classes"], include_other=config["include_other"])
    total_counts = dict(Counter(item.label for item in items))
    if args.limit_per_category:
        groups = defaultdict(list)
        for item in items:
            groups[item.label].append(item)
        rng = random.Random(config["seed"])
        items = [item for group in groups.values() for item in rng.sample(group, min(len(group), args.limit_per_category))]
    name = args.output_name or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        parser.error("--output-name: use letters, digits, underscore or hyphen")
    output = config["output_dir"] / name
    output.mkdir(parents=True, exist_ok=False)
    serial_config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    records = [{"path": str(item.source), "source_label": item.label, "sha256": item.sha256} for item in items]
    write_json(output / "run.json", {"config": serial_config, "source": str(source),
               "selected_models": [spec["name"] for spec in selected], "invalid_images": invalid,
               "available_source_counts": total_counts, "evaluated_source_counts": dict(Counter(i.label for i in items)),
               "smoke_test": bool(args.limit_per_category), "images": records})
    print(f"Images: {len(items)}, invalid: {invalid}, categories: {Counter(i.label for i in items)}", flush=True)
    print(f"Output: {output}", flush=True)
    results, errors = [], {}
    for spec in selected:
        model_dir = output / spec["name"]
        model_dir.mkdir()
        job = model_dir / "job.json"
        raw_path = model_dir / "detections.json"
        write_json(job, {"config": serial_config, "model": spec, "images": records, "result": str(raw_path)})
        interpreter = str(ROOT / spec["python"]) if "python" in spec else sys.executable
        print(f"Running {spec['name']} ... log: {model_dir / 'inference.log'}", flush=True)
        try:
            native_floor = {"picodet": 0.025, "nanodet": 0.05}.get(spec["adapter"], 0)
            if min(*config["verification_thresholds"].values(), config["other_threshold"]) < native_floor:
                raise ValueError(f"Decision threshold below {spec['adapter']} native score floor {native_floor}")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            with (model_dir / "inference.log").open("w", encoding="utf-8") as log:
                process = subprocess.run([interpreter, str(ROOT / "scripts/detect_worker.py"), str(job)],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"Worker exit {process.returncode}; see {spec['name']}/inference.log")
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            results.append(render_model(raw, config, model_dir))
            print(f"{spec['name']}: accuracy={results[-1]['accuracy']:.2%}", flush=True)
        except Exception as exc:
            errors[spec["name"]] = str(exc)
            print(f"FAILED {spec['name']}: {exc}", flush=True)
        render_summary(results, errors, output)
    print(f"Report: {output / 'report.html'}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
