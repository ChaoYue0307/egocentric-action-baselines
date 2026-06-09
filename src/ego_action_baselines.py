from __future__ import annotations

import csv
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


@dataclass
class WindowSample:
    start: int
    end: int
    center: int
    label: str
    label_fraction: float


def load_caption(annotation: Path) -> dict:
    with h5py.File(annotation, "r") as h5:
        raw = h5["caption"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def load_frame_numbers(annotation: Path) -> np.ndarray:
    with h5py.File(annotation, "r") as h5:
        if "slam/frame_names" in h5:
            names = []
            for item in h5["slam/frame_names"]:
                text = np.asarray(item).tobytes().decode("utf-8", errors="replace").strip("\x00")
                stem = text.rsplit(".", 1)[0] if "." in text else text
                names.append(int(stem))
            return np.asarray(names, dtype=np.int64)
        return np.asarray(h5["video/frame_number"][...], dtype=np.int64)


def load_hand_joints(annotation: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(annotation, "r") as h5:
        left = np.asarray(h5["hand_mocap/left_joints_3d"][...], dtype=np.float32)
        right = np.asarray(h5["hand_mocap/right_joints_3d"][...], dtype=np.float32)
    return left, right


def frame_labels_from_caption(caption: dict, frame_numbers: np.ndarray, target: str) -> list[str]:
    labels = [""] * len(frame_numbers)
    segments = caption.get("segments", [])
    for segment in segments:
        subtask = str(segment.get("Sub Task", "")).strip()
        if target == "subtask":
            start = int(segment.get("start_frame", 0))
            end = int(segment.get("end_frame", 0))
            mask = (frame_numbers >= start) & (frame_numbers <= end)
            for idx in np.flatnonzero(mask):
                labels[int(idx)] = subtask
            continue

        for action in segment.get("Current Action", []):
            label = str(action.get("label", "")).strip()
            if not label or label.upper() == "N/A":
                continue
            start = int(action.get("start_frame", segment.get("start_frame", 0)))
            end = int(action.get("end_frame", segment.get("end_frame", 0)))
            mask = (frame_numbers >= start) & (frame_numbers <= end)
            for idx in np.flatnonzero(mask):
                labels[int(idx)] = label
    return labels


def majority_label(labels: list[str], min_fraction: float) -> tuple[str, float]:
    valid = [x for x in labels if x]
    if not valid:
        return "", 0.0
    label, count = Counter(valid).most_common(1)[0]
    fraction = count / len(valid)
    return (label, fraction) if fraction >= min_fraction else ("", fraction)


def build_windows(labels: list[str], window_frames: int, stride_frames: int, min_label_fraction: float, max_windows: int | None) -> list[WindowSample]:
    windows: list[WindowSample] = []
    for start in range(0, len(labels) - window_frames + 1, stride_frames):
        end = start + window_frames
        label, fraction = majority_label(labels[start:end], min_label_fraction)
        if label:
            windows.append(WindowSample(start=start, end=end, center=(start + end - 1) // 2, label=label, label_fraction=fraction))
        if max_windows and len(windows) >= max_windows:
            break
    return windows


def temporal_stats(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).reshape(arr.shape[0], -1)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    vel = np.diff(flat, axis=0) if len(flat) > 1 else np.zeros_like(flat)
    return np.concatenate([
        flat.mean(axis=0),
        flat.std(axis=0),
        flat.min(axis=0),
        flat.max(axis=0),
        flat[-1] - flat[0],
        vel.mean(axis=0),
        vel.std(axis=0),
    ]).astype(np.float32)


def hand_features(left: np.ndarray, right: np.ndarray, windows: list[WindowSample]) -> np.ndarray:
    feats = []
    for window in windows:
        hands = np.concatenate([left[window.start:window.end], right[window.start:window.end]], axis=1)
        root = hands[:, :1, :]
        feats.append(temporal_stats(hands - root))
    return np.stack(feats).astype(np.float32)


def rgb_frame_feature(frame: np.ndarray, grid_size: int = 8, hist_bins: int = 8) -> np.ndarray:
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = rgb.reshape(-1, 3).mean(axis=0)
    std = rgb.reshape(-1, 3).std(axis=0)
    hists = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=hist_bins, range=(0, 1))
        hist = hist.astype(np.float32)
        hists.append(hist / max(float(hist.sum()), 1.0))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grid = cv2.resize(gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA).reshape(-1)
    gx, gy = np.gradient(gray)
    edge = np.asarray([np.abs(gx).mean(), np.abs(gy).mean(), np.abs(gx).std(), np.abs(gy).std()], dtype=np.float32)
    return np.concatenate([mean, std, *hists, grid, edge]).astype(np.float32)


def rgb_features(video_path: Path, windows: list[WindowSample]) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    needed = sorted({idx for window in windows for idx in (window.start, window.center, window.end - 1)})
    feature_by_idx: dict[int, np.ndarray] = {}
    feats = []
    try:
        wanted = set(needed)
        current = 0
        ok = True
        while ok and wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if current in wanted:
                feature_by_idx[current] = rgb_frame_feature(frame)
                wanted.remove(current)
            current += 1
        missing = sorted(wanted)
        if missing:
            raise RuntimeError(f"Could not read required video frames: {missing[:5]}")
        for window in windows:
            frame_feats = [feature_by_idx[idx] for idx in (window.start, window.center, window.end - 1)]
            feats.append(temporal_stats(np.stack(frame_feats)))
    finally:
        cap.release()
    return np.stack(feats).astype(np.float32)


def encode_labels(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    seen: OrderedDict[str, int] = OrderedDict()
    for label in labels:
        if label not in seen:
            seen[label] = len(seen)
    return np.asarray([seen[label] for label in labels], dtype=np.int64), list(seen.keys())


def stratified_split(y: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, test = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        if len(idx) < 2:
            train.extend(idx.tolist())
            continue
        n_test = max(1, min(len(idx) - 1, int(round(len(idx) * test_fraction))))
        test.extend(idx[:n_test].tolist())
        train.extend(idx[n_test:].tolist())
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def chronological_split(y: np.ndarray, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split windows by time order to reduce leakage from overlapping windows."""
    n = len(y)
    if n < 2:
        return np.arange(n, dtype=np.int64), np.asarray([], dtype=np.int64)
    n_test = max(1, min(n - 1, int(round(n * test_fraction))))
    split = n - n_test
    return np.arange(split, dtype=np.int64), np.arange(split, n, dtype=np.int64)


def make_split(y: np.ndarray, test_fraction: float, seed: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "chronological":
        return chronological_split(y, test_fraction)
    if strategy == "stratified":
        return stratified_split(y, test_fraction, seed)
    raise ValueError(f"Unknown split strategy: {strategy}")


def fit_scaler(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((X - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, n_classes: int, epochs: int, lr: float, l2: float, seed: int):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(X.shape[1], n_classes)).astype(np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
    class_weights = len(train_idx) / np.maximum(counts, 1.0) / max(n_classes, 1)
    sample_weights = class_weights[y[train_idx]]
    sample_weights = sample_weights / max(float(sample_weights.mean()), 1e-6)
    Y = np.eye(n_classes, dtype=np.float32)[y[train_idx]]
    history = []
    for epoch in range(1, epochs + 1):
        logits = X[train_idx] @ W + b
        probs = softmax(logits)
        diff = (probs - Y) * sample_weights[:, None] / len(train_idx)
        W -= lr * (X[train_idx].T @ diff + l2 * W)
        b -= lr * diff.sum(axis=0)
        if epoch == 1 or epoch == epochs:
            history.append({"epoch": epoch, "train_accuracy": float(np.mean(probs.argmax(axis=1) == y[train_idx]))})
    return W, b, history


def train_torch_mlp(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    hidden_dim: int,
) -> tuple[np.ndarray, list[dict], object]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for --model mlp. Install with: pip install -e '.[mlp]'") from exc

    torch.manual_seed(seed)
    X_train = torch.as_tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(y[train_idx], dtype=torch.long)
    X_test = torch.as_tensor(X[test_idx], dtype=torch.float32)
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
    weights = len(train_idx) / np.maximum(counts, 1.0) / max(n_classes, 1)
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.15),
        torch.nn.Linear(hidden_dim, n_classes),
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == epochs:
            pred = logits.detach().argmax(dim=1)
            history.append({
                "epoch": epoch,
                "train_accuracy": float((pred == y_train).float().mean().item()),
                "train_loss": float(loss.detach().item()),
            })
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_test), dim=1).cpu().numpy()
    return probs.astype(np.float32), history, model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> tuple[dict, list[dict], np.ndarray]:
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[int(t), int(p)] += 1
    rows, f1s, recalls, weighted = [], [], [], 0.0
    total = int(cm.sum())
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            recalls.append(recall)
            f1s.append(f1)
            weighted += f1 * support
        rows.append({"class_id": i, "class_name": name, "support": support, "predicted": predicted, "precision": precision, "recall": recall, "f1": f1})
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_f1": float(weighted / total) if total else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "num_eval": int(len(y_true)),
        "num_classes": len(class_names),
    }, rows, cm


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(name: str, X: np.ndarray, y: np.ndarray, class_names: list[str], windows: list[WindowSample], out_dir: Path, args, model_type: str = "softmax") -> dict:
    train_idx, test_idx = make_split(y, args.test_fraction, args.seed, args.split_strategy)
    Xs, mean, std = fit_scaler(X, train_idx)
    if model_type == "softmax":
        W, b, history = train_softmax(Xs, y, train_idx, len(class_names), args.epochs, args.learning_rate, args.l2, args.seed)
        probs = softmax(Xs[test_idx] @ W + b)
        model_payload = {"mean": mean, "std": std, "W": W, "b": b}
    elif model_type == "mlp":
        probs, history, model = train_torch_mlp(Xs, y, train_idx, test_idx, len(class_names), args.epochs, args.learning_rate, args.l2, args.seed, args.mlp_hidden_dim)
        model_payload = {"mean": mean, "std": std, "torch_model": model}
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    pred = probs.argmax(axis=1)
    metrics, per_class, cm = compute_metrics(y[test_idx], pred, class_names)
    metrics.update({"experiment": name, "model": model_type, "feature_dim": int(X.shape[1]), "num_windows": int(len(y)), "num_train": int(len(train_idx)), "num_test": int(len(test_idx)), "split_strategy": args.split_strategy})
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (exp_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_csv(exp_dir / "per_class_metrics.csv", per_class, ["class_id", "class_name", "support", "predicted", "precision", "recall", "f1"])
    with (exp_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true\\pred"] + class_names)
        for i, cname in enumerate(class_names):
            writer.writerow([cname] + [int(x) for x in cm[i]])
    pred_rows = []
    for k, idx in enumerate(test_idx):
        w = windows[int(idx)]
        pred_rows.append({
            "window_index": int(idx),
            "start_frame": w.start,
            "end_frame": w.end - 1,
            "true_label": class_names[int(y[idx])],
            "predicted_label": class_names[int(pred[k])],
            "confidence": float(probs[k, pred[k]]),
            "correct": int(pred[k] == y[idx]),
        })
    write_csv(exp_dir / "predictions.csv", pred_rows, ["window_index", "start_frame", "end_frame", "true_label", "predicted_label", "confidence", "correct"])
    if model_type == "softmax":
        np.savez_compressed(exp_dir / "model.npz", **model_payload, class_names=np.asarray(class_names, dtype=object))
    else:
        import torch

        torch.save({
            "state_dict": model_payload["torch_model"].state_dict(),
            "mean": model_payload["mean"],
            "std": model_payload["std"],
            "class_names": class_names,
            "hidden_dim": args.mlp_hidden_dim,
        }, exp_dir / "model.pt")
    return metrics


def build_dataset(data_root: Path, video_name: str, target: str, window_frames: int, stride_frames: int, min_label_fraction: float, max_windows: int | None):
    annotation = data_root / "annotation.hdf5"
    caption = load_caption(annotation)
    frame_numbers = load_frame_numbers(annotation)
    labels = frame_labels_from_caption(caption, frame_numbers, target)
    windows = build_windows(labels, window_frames, stride_frames, min_label_fraction, max_windows)
    y, class_names = encode_labels([w.label for w in windows])
    left, right = load_hand_joints(annotation)
    X_hand = hand_features(left, right, windows)
    X_rgb = rgb_features(data_root / video_name, windows)
    return windows, y, class_names, X_rgb, X_hand
