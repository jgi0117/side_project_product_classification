from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_ALIASES = {
    "book": "book",
    "books": "book",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "bycicle": "bicycle",
    "bycicles": "bicycle",
    "bike": "bicycle",
    "guitar": "guitar",
    "guitars": "guitar",
    "laptop": "laptop",
    "laptops": "laptop",
}


@dataclass(frozen=True)
class ImageItem:
    source: Path
    label: str
    sha256: str


def _label_from_path(path: Path, root: Path) -> str | None:
    label = None
    for part in path.relative_to(root).parts[:-1]:
        normalized = part.strip().lower().replace(" ", "_")
        label = CLASS_ALIASES.get(normalized, label)
    return label


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(
    source: Path, expected_classes: list[str]
) -> tuple[list[ImageItem], int]:
    """Read mounted images in place and infer labels from category folders."""
    if not source.exists():
        raise FileNotFoundError(f"원본 데이터 경로가 없습니다: {source}")

    items: list[ImageItem] = []
    bad_images = 0
    seen_hashes: dict[str, str] = {}
    paths = sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in tqdm(paths, desc="Validating Drive images", unit="image"):
        label = _label_from_path(path, source)
        if label not in expected_classes:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            digest = _sha256(path)
        except (OSError, ValueError):
            bad_images += 1
            continue
        previous_label = seen_hashes.get(digest)
        if previous_label is not None:
            if previous_label != label:
                raise ValueError(
                    f"같은 이미지가 서로 다른 클래스에 있습니다: {path}"
                )
            continue
        seen_hashes[digest] = label
        items.append(ImageItem(path.resolve(), label, digest))

    counts = Counter(item.label for item in items)
    missing = [name for name in expected_classes if counts[name] == 0]
    if missing:
        raise ValueError(
            f"클래스 폴더를 찾지 못했습니다: {missing}. "
            f"지원 이름: {sorted(CLASS_ALIASES)}"
        )
    return items, bad_images
