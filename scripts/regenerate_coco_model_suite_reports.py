from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_coco_model_suite import (  # noqa: E402
    finalize_model_report,
    resolve_path,
    write_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="기존 COCO model suite 예측으로 그래프와 집계 다시 생성"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "coco_models.yaml")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+", help="모델 id 일부만 재생성")
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k는 1 이상이어야 합니다.")

    with args.config.resolve().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base_output_dir = resolve_path(config["output_dir"])
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else base_output_dir
        if args.top_k == 1
        else base_output_dir.with_name(f"{base_output_dir.name}_top{args.top_k}")
    )
    selected = [
        model for model in config["models"] if not args.models or model["id"] in args.models
    ]
    unknown = set(args.models or []) - {model["id"] for model in selected}
    if unknown:
        raise SystemExit(f"설정에 없는 모델입니다: {sorted(unknown)}")

    metrics = []
    for model in selected:
        model_dir = output_dir / model["id"]
        predictions = model_dir / "raw_predictions.csv"
        if not predictions.is_file():
            print(f"[{model['id']}] 건너뜀: {predictions} 없음", file=sys.stderr)
            continue
        metrics.append(
            finalize_model_report(
                model_dir,
                list(config["classes"]),
                dict(config["target_model_labels"]),
                dict(config["verification_thresholds"]),
                args.top_k,
            )
        )
        print(f"[{model['id']}] 보고서 재생성 완료")

    write_comparison(output_dir, metrics)
    print(f"완료: {len(metrics)}/{len(selected)} models")
    print(f"결과: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
