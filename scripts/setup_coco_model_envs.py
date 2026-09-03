from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / ".model_envs"
THIRD_PARTY = ROOT / "third_party"


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def ensure_repo(url: str, destination: Path, env: dict[str, str]) -> None:
    if (destination / ".git").is_dir():
        print(f"기존 저장소 확인: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", url, str(destination)], env=env)


def ensure_venv(name: str, python_version: str, env: dict[str, str]) -> Path:
    destination = ENV_ROOT / name
    python = destination / "Scripts" / "python.exe"
    if not python.is_file():
        run(["uv", "venv", "--python", python_version, str(destination)], env=env)
    return python


def uv_install(
    python: Path,
    packages: list[str],
    env: dict[str, str],
    *options: str,
) -> None:
    run(
        ["uv", "pip", "install", "--python", str(python), *options, *packages],
        env=env,
    )


def setup_yolox(env: dict[str, str]) -> None:
    repository = THIRD_PARTY / "YOLOX"
    ensure_repo("https://github.com/Megvii-BaseDetection/YOLOX.git", repository, env)
    python = ensure_venv("yolox", "3.11", env)
    uv_install(
        python,
        [
            "numpy<2",
            "torch==2.2.2",
            "torchvision==0.17.2",
            "opencv-python",
            "loguru",
            "tqdm",
            "tabulate",
            "thop",
            "ninja",
            "psutil",
            "packaging",
            "pycocotools",
        ],
        env,
    )
    # YOLOX의 추론에는 오래된 onnx-simplifier 빌드 의존성이 필요하지 않습니다.
    uv_install(python, [str(repository)], env, "--no-deps", "-e")


def setup_nanodet(env: dict[str, str]) -> None:
    repository = THIRD_PARTY / "nanodet"
    ensure_repo("https://github.com/RangiLyu/nanodet.git", repository, env)
    python = ensure_venv("nanodet", "3.9", env)
    uv_install(
        python,
        [
            "numpy<2",
            "setuptools==69.5.1",
            "torch==1.13.1",
            "torchvision==0.14.1",
            "opencv-python",
            "PyYAML",
            "termcolor",
            "tensorboard",
            "tqdm",
            "omegaconf>=2.0.1",
            "pytorch-lightning==1.9.5",
            "torchmetrics==0.11.4",
            "imagesize",
            "pyaml",
            "tabulate",
            "matplotlib",
            "pycocotools",
            "-e",
            str(repository),
        ],
        env,
    )


def setup_rtmdet(env: dict[str, str]) -> None:
    python = ensure_venv("rtmdet", "3.11", env)
    uv_install(
        python,
        [
            "setuptools==69.5.1",
            "wheel",
            "ninja",
            "cython",
            "numpy<2",
            "torch==2.1.2",
            "torchvision==0.16.2",
            "mmengine==0.10.7",
            "mmdet==3.3.0",
        ],
        env,
    )
    # mmcv 2.1은 빌드 요구사항에 pkg_resources를 선언하지 않아 격리를 끕니다.
    uv_install(python, ["mmcv==2.1.0"], env, "--no-build-isolation")


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO detector별 격리 실행 환경 설치")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rtmdet", "nanodet", "yolox"],
        default=["rtmdet", "nanodet", "yolox"],
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(ROOT / ".uv-cache")
    installers = {
        "rtmdet": setup_rtmdet,
        "nanodet": setup_nanodet,
        "yolox": setup_yolox,
    }
    failures = []
    for model in args.models:
        print(f"[{model}] 환경 준비 시작", flush=True)
        try:
            installers[model](env)
        except subprocess.CalledProcessError as error:
            failures.append((model, error.returncode))
            print(f"[{model}] 설치 실패(exit={error.returncode})", file=sys.stderr)
    if failures:
        raise SystemExit(f"환경 설치 실패: {failures}")


if __name__ == "__main__":
    main()
