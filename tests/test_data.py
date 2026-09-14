from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.data import discover_images  # noqa: E402


def test_discover_uses_actual_folder_layout_without_copying(tmp_path: Path) -> None:
    source = tmp_path / "sample"
    for class_index, class_name in enumerate(("book", "bycicle", "guitar", "laptop", "tv")):
        child_folder = "public" if class_name == "laptop" else "open_image"
        folder = source / class_name / child_folder
        folder.mkdir(parents=True, exist_ok=True)
        for image_index in range(3):
            Image.new(
                "RGB", (8, 8), (image_index * 60, class_index * 60, 10)
            ).save(
                folder / f"{image_index}.jpg"
            )

    items, invalid = discover_images(
        source, ["computer", "book"]
    )

    assert invalid == 0
    assert len(items) == 15
    assert sorted({item.label for item in items}) == [
        "bicycle",
        "book",
        "computer",
        "guitar",
    ]
    assert not (tmp_path / "processed").exists()
