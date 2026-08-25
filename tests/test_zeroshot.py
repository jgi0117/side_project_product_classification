from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.benchmark import _resolve_imagenet_indices  # noqa: E402


def test_imagenet_patterns_resolve_four_target_classes() -> None:
    names = {
        0: "bicycle-built-for-two",
        1: "mountain bike, all-terrain bike, off-roader",
        2: "book_jacket, dust_cover, dust_jacket, dust_wrapper",
        3: "comic_book",
        4: "acoustic_guitar",
        5: "electric_guitar",
        6: "laptop, laptop computer",
        7: "notebook, notebook computer",
    }
    patterns = {
        "bicycle": ["bicycle-built-for-two", "mountain bike", "all-terrain bike"],
        "book": ["book jacket", "comic book"],
        "guitar": ["acoustic guitar", "electric guitar"],
        "laptop": ["laptop", "notebook computer"],
    }
    indices, matched = _resolve_imagenet_indices(
        names, ["bicycle", "book", "guitar", "laptop"], patterns
    )
    assert indices == {
        "bicycle": [0, 1],
        "book": [2, 3],
        "guitar": [4, 5],
        "laptop": [6, 7],
    }
    assert matched["laptop"] == [
        "laptop, laptop computer",
        "notebook, notebook computer",
    ]
    assert matched["book"] == [
        "book jacket, dust cover, dust jacket, dust wrapper",
        "comic book",
    ]
