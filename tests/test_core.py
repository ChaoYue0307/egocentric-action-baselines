from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters import XperienceActionAdapter  # noqa: E402
from ego_action_baselines import (  # noqa: E402
    WindowSample,
    chronological_split,
    compute_metrics,
    frame_labels_from_caption,
    majority_label,
    make_split,
    predict_majority,
    run_experiment,
    softmax,
)
from ego_action_baselines_cli import aggregate_summaries, model_types_for  # noqa: E402


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


def test_run_experiment_rejects_unknown_model(tmp_path: Path) -> None:
    args = Namespace(test_fraction=0.5, seed=0, split_strategy="chronological", epochs=1, learning_rate=0.1, l2=0.0, mlp_hidden_dim=8)
    windows = [WindowSample(0, 1, 0, "a", 1.0), WindowSample(1, 2, 1, "b", 1.0)]
    X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.asarray([0, 1], dtype=np.int64)
    try:
        run_experiment("bad", X, y, ["a", "b"], windows, tmp_path, args, "unknown")
    except ValueError as exc:
        assert "Unknown model type" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_xperience_action_adapter_paths(tmp_path: Path) -> None:
    adapter = XperienceActionAdapter(tmp_path)
    assert adapter.annotation_path == tmp_path / "annotation.hdf5"
    assert adapter.video_path == tmp_path / "fisheye_cam0.mp4"
    assert "hand_joints_3d" in adapter.describe()["signals"]


def test_majority_baseline_predictions() -> None:
    y = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    probs, history = predict_majority(y, np.asarray([0, 1, 2, 3]), np.asarray([4]), 2)
    assert probs.argmax(axis=1).tolist() == [0]
    assert history[0]["majority_class_id"] == 0


def test_model_types_and_aggregate() -> None:
    assert model_types_for("all") == ["majority", "softmax", "mlp"]
    aggregate = aggregate_summaries([
        {"experiments": {"rgb_only": {"accuracy": 0.5, "macro_f1": 0.4}}},
        {"experiments": {"rgb_only": {"accuracy": 1.0, "macro_f1": 0.8}}},
    ])
    assert aggregate["num_episodes"] == 2
    assert aggregate["experiments"]["rgb_only"]["accuracy_mean"] == 0.75
