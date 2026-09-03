from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(path: Path, spec: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    minimum = int(spec.get("min_bytes", 1024 * 1024))
    if path.stat().st_size < minimum:
        raise ValueError(
            f"다운로드 파일이 너무 작습니다: {path} "
            f"({path.stat().st_size:,} < {minimum:,} bytes)"
        )
    with path.open("rb") as handle:
        signature = handle.read(256).lstrip().lower()
    if signature.startswith((b"<!doctype html", b"<html")):
        raise ValueError(f"모델 대신 HTML 응답을 받았습니다: {path}")
    expected_prefix = str(spec.get("sha256_prefix", "")).lower()
    if expected_prefix:
        actual = _sha256(path)
        if not actual.startswith(expected_prefix):
            raise ValueError(
                f"SHA-256 불일치: {path.name} "
                f"(expected prefix={expected_prefix}, actual={actual})"
            )


def ensure_download(spec: dict[str, Any], root: Path) -> tuple[Path, bool]:
    destination = Path(spec["destination"])
    if not destination.is_absolute():
        destination = root / destination
    if destination.exists():
        _validate(destination, spec)
        return destination, False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        str(spec["url"]),
        headers={"User-Agent": "Mozilla/5.0 model-benchmark-downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        _validate(temporary, spec)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination, True
