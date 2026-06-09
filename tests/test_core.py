from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego_action_baselines import (  # noqa: E402
    chronological_split,
    compute_metrics,
    frame_labels_from_caption,
    majority_label,
    make_split,
    softmax,
)


def test_majority_label_respects_fraction() -> None:
    assert majority_label(["pour", "pour", "pick"], 0.6) == ("pour", 2 / 3)
    assert majority_label(["pour", "pick", "pick"], 0.8) == ("", 2 / 3)


def test_frame_labels_from_caption_action_ranges() -> None:
    caption = {
        "segments": [
            {
                "Current Action": [
                    {"label": "Pick up kettle", "start_frame": 10, "end_frame": 20},
                    {"label": "Pour water", "start_frame": 30, "end_frame": 40},
                ]
            }
        ]
    }
    labels = frame_labels_from_caption(caption, np.asarray([5, 10, 15, 25, 30, 35, 45]), "action")
    assert labels == ["", "Pick up kettle", "Pick up kettle", "", "Pour water", "Pour water", ""]


def test_chronological_split_keeps_tail_for_eval() -> None:
    train, test = chronological_split(np.arange(10), 0.3)
    assert train.tolist() == list(range(7))
    assert test.tolist() == [7, 8, 9]


def test_make_split_rejects_unknown_strategy() -> None:
    try:
        make_split(np.arange(4), 0.25, 0, "unknown")
    except ValueError as exc:
        assert "Unknown split strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_softmax_rows_sum_to_one() -> None:
    probs = softmax(np.asarray([[1.0, 2.0], [3.0, 1.0]], dtype=np.float32))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 1] > probs[0, 0]


def test_compute_metrics_macro_f1() -> None:
    metrics, rows, cm = compute_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 1]), ["a", "b"])
    assert metrics["accuracy"] == 0.75
    assert round(metrics["macro_f1"], 3) == 0.733
    assert rows[0]["class_name"] == "a"
    assert cm.tolist() == [[1, 1], [0, 2]]
