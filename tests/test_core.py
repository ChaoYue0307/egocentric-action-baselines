from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters import XperienceActionAdapter  # noqa: E402
from ego_action_baselines import (  # noqa: E402
    WindowSample,
    blocked_instance_split,
    calibration_report,
    chronological_split,
    compute_metrics,
    encode_labels,
    frame_labels_from_caption,
    grouped_segment_split,
    majority_label,
    make_split,
    predict_majority,
    run_experiment,
    softmax,
    top_confusions,
    train_gated_fusion,
    window_overlap_purge,
)
from ego_action_baselines_cli import aggregate_summaries, model_types_for, run_seed_pass  # noqa: E402


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
    windows = [WindowSample(i, i + 1, i, "a", 1.0) for i in range(4)]
    try:
        make_split(windows, np.arange(4), 0.25, 0, "unknown")
    except ValueError as exc:
        assert "Unknown split strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_grouped_segment_split_keeps_train_support_for_test_classes() -> None:
    labels = ["a"] * 4 + ["b"] * 4 + ["a"] * 4 + ["b"] * 4
    windows = [WindowSample(i * 4, i * 4 + 4, i * 4 + 1, label, 1.0) for i, label in enumerate(labels)]
    y, _ = encode_labels([w.label for w in windows])
    train, test = grouped_segment_split(windows, y, 0.25, 0)
    assert len(test) > 0
    assert set(y[test].tolist()) <= set(y[train].tolist())
    assert len(set(train.tolist()) & set(test.tolist())) == 0


def test_grouped_segment_split_raises_on_single_instances() -> None:
    labels = ["a"] * 4 + ["b"] * 4
    windows = [WindowSample(i * 4, i * 4 + 4, i * 4 + 1, label, 1.0) for i, label in enumerate(labels)]
    y, _ = encode_labels([w.label for w in windows])
    try:
        grouped_segment_split(windows, y, 0.25, 0)
    except ValueError as exc:
        assert "blocked-instance" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_blocked_instance_split_holds_out_instance_tails() -> None:
    labels = ["a"] * 8 + ["b"] * 8
    windows = [WindowSample(i * 4, i * 4 + 4, i * 4 + 1, label, 1.0) for i, label in enumerate(labels)]
    y, _ = encode_labels([w.label for w in windows])
    train, test = blocked_instance_split(windows, y, 0.25)
    assert test.tolist() == [6, 7, 14, 15]
    assert set(y[test].tolist()) <= set(y[train].tolist())


def test_window_overlap_purge_drops_shared_frames() -> None:
    windows = [WindowSample(0, 20, 9, "a", 1.0), WindowSample(10, 30, 19, "a", 1.0), WindowSample(40, 60, 49, "b", 1.0)]
    kept = window_overlap_purge(windows, np.asarray([0, 2]), np.asarray([1]))
    assert kept.tolist() == [2]


def test_top_confusions_ranks_off_diagonal() -> None:
    cm = np.asarray([[5, 3, 0], [1, 6, 0], [0, 4, 2]], dtype=np.int64)
    rows = top_confusions(cm, ["a", "b", "c"], k=2)
    assert rows[0] == {"true": "c", "predicted": "b", "count": 4, "fraction_of_true": round(4 / 6, 4)}
    assert rows[1]["count"] == 3
    assert all(r["true"] != r["predicted"] for r in rows)


def test_gated_fusion_trains_and_keeps_gate_soft() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n, nc = 40, 3
    y = np.tile(np.arange(nc), n // nc + 1)[:n].astype(np.int64)
    X_rgb = rng.normal(size=(n, 6)).astype(np.float32) + y[:, None]
    X_hand = rng.normal(size=(n, 5)).astype(np.float32) + y[:, None]
    train_idx = np.arange(0, n, 2)
    test_idx = np.arange(1, n, 2)
    probs, history = train_gated_fusion(X_rgb, X_hand, y, train_idx, test_idx, nc, epochs=40, lr=5e-3, l2=1e-3, seed=0)
    assert probs.shape == (len(test_idx), nc)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    gate = history[-1]["test_mean_rgb_gate"]
    assert 0.05 < gate < 0.95


def test_calibration_report_expected_calibration_error() -> None:
    probs = np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    report = calibration_report(probs, np.asarray([0, 1]), n_bins=10)
    assert round(report["ece"], 2) == 0.45
    assert sum(b["count"] for b in report["bins"]) == 2


def test_late_fusion_averages_probabilities(tmp_path: Path) -> None:
    args = Namespace(test_fraction=0.5, seed=0, split_strategy="stratified", purge_overlap=False, epochs=5, learning_rate=0.1, l2=0.0, mlp_hidden_dim=8, model="softmax")
    labels = ["a" if i % 2 == 0 else "b" for i in range(8)]
    windows = [WindowSample(i * 10, i * 10 + 10, i * 10 + 5, label, 1.0) for i, label in enumerate(labels)]
    y, class_names = encode_labels([w.label for w in windows])
    X = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    datasets = {"rgb_only": X, "hand_joints_only": X.copy(), "rgb_hand_fusion": np.concatenate([X, X], axis=1).astype(np.float32)}
    results = run_seed_pass(args, datasets, y, class_names, windows, tmp_path, 0)
    assert "rgb_hand_late_fusion" in results
    expected = (results["rgb_only"]["probs"] + results["hand_joints_only"]["probs"]) / 2.0
    assert np.allclose(results["rgb_hand_late_fusion"]["probs"], expected)
    assert (tmp_path / "rgb_hand_late_fusion" / "calibration.json").exists()


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
    assert model_types_for("classical") == ["majority", "softmax"]
    assert model_types_for("all") == ["majority", "softmax", "mlp"]
    aggregate = aggregate_summaries([
        {"experiments": {"rgb_only": {"accuracy": 0.5, "macro_f1": 0.4}}},
        {"experiments": {"rgb_only": {"accuracy": 1.0, "macro_f1": 0.8}}},
    ])
    assert aggregate["num_episodes"] == 2
    assert aggregate["experiments"]["rgb_only"]["accuracy_mean"] == 0.75
