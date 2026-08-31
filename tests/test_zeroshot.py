from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_benchmark.benchmark import (  # noqa: E402
    _apply_top_k_thresholds,
    _primary_label_operating_metrics,
    _resolve_imagenet_indices,
)


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


def test_primary_label_metrics_do_not_call_additional_objects_false_positives() -> None:
    classes = ["bicycle", "book", "guitar", "laptop"]
    targets = [0, 1, 2, 3]
    accepted = np.asarray(
        [
            [True, False, False, False],
            [False, True, False, True],
            [False, False, True, False],
            [False, False, False, True],
        ]
    )

    metrics = _primary_label_operating_metrics(targets, accepted, classes)

    assert metrics["expected_label_accept_rate_macro"] == 1.0
    assert metrics["multiple_label_rate"] == 0.25
    assert metrics["false_accept_rate_available"] is False
    assert (
        metrics["per_class_primary_label_metrics"]["book"][
            "additional_label_activation_rate"
        ]
        == 1.0
    )


def test_top_k_one_accepts_at_most_one_threshold_passing_label() -> None:
    scores = np.asarray([[0.20, 0.18, 0.01, 0.10], [0.04, 0.03, 0.02, 0.01]])
    thresholds = np.asarray([0.05, 0.05, 0.05, 0.05])

    accepted = _apply_top_k_thresholds(scores, thresholds, top_k=1)

    assert accepted.tolist() == [
        [True, False, False, False],
        [False, False, False, False],
    ]
