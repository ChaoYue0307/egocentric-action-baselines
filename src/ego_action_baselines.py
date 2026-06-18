from __future__ import annotations

import csv
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


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


def make_dino_embedder(model_name: str = "dinov2_vits14"):
    """Frozen DINOv2 frame embedder. Downloads hub weights on first use."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for --rgb-embedding dino. Install with: pip install -e '.[dino]'") from exc

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()

    def embed(frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
        with torch.no_grad():
            feat = model(tensor)
        return np.asarray(feat[0], dtype=np.float32)

    return embed


def rgb_features(video_path: Path, windows: list[WindowSample], frame_feature_fn=None) -> np.ndarray:
    frame_feature_fn = frame_feature_fn or rgb_frame_feature
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
                feature_by_idx[current] = frame_feature_fn(frame)
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


def label_run_groups(windows: list[WindowSample]) -> np.ndarray:
    """Group windows into contiguous same-label runs, approximating action instances."""
    groups = np.zeros(len(windows), dtype=np.int64)
    gid = 0
    for i, window in enumerate(windows):
        if i and window.label != windows[i - 1].label:
            gid += 1
        groups[i] = gid
    return groups


def grouped_segment_split(windows: list[WindowSample], y: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole action instances so every test class keeps train support.

    Classes with a single instance stay fully in train; at least one instance per
    class always remains in train.
    """
    groups = label_run_groups(windows)
    rng = np.random.default_rng(seed)
    test: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        cls_groups = np.unique(groups[cls_idx])
        if len(cls_groups) < 2:
            continue
        shuffled = cls_groups.copy()
        rng.shuffle(shuffled)
        target = max(1, int(round(len(cls_idx) * test_fraction)))
        picked: list[int] = []
        count = 0
        for group in shuffled:
            if count >= target or len(picked) >= len(cls_groups) - 1:
                break
            members = np.flatnonzero(groups == group)
            picked.append(int(group))
            count += len(members)
        test.extend(np.flatnonzero(np.isin(groups, picked)).tolist())
    test_set = sorted(set(test))
    if not test_set:
        raise ValueError(
            "grouped-segment needs at least one class with two or more action instances, "
            "but every class in this data occurs as a single instance. Use blocked-instance "
            "for within-episode evaluation, or pass more episodes via --data-roots."
        )
    train = np.asarray([i for i in range(len(y)) if i not in set(test_set)], dtype=np.int64)
    return train, np.asarray(test_set, dtype=np.int64)


def blocked_instance_split(windows: list[WindowSample], y: np.ndarray, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the chronological tail of each action instance.

    Measures within-instance generalization: every test class keeps train
    support, and combined with overlap purging no frame is shared between train
    and test. Classes whose train support would be fully purged stay in train.
    """
    groups = label_run_groups(windows)
    test: list[int] = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        if len(members) < 2:
            continue
        n_test = min(max(1, int(round(len(members) * test_fraction))), len(members) - 1)
        test.extend(int(i) for i in members[-n_test:])
    test_set = set(test)
    train = np.asarray([i for i in range(len(y)) if i not in test_set], dtype=np.int64)
    test_idx = np.asarray(sorted(test_set), dtype=np.int64)
    purged_train = window_overlap_purge(windows, train, test_idx)
    supported = set(y[purged_train].tolist())
    moved_back = [int(i) for i in test_idx if int(y[i]) not in supported]
    if moved_back:
        test_idx = np.asarray([int(i) for i in test_idx if int(i) not in set(moved_back)], dtype=np.int64)
        train = np.asarray(sorted(set(train.tolist()) | set(moved_back)), dtype=np.int64)
    return train, test_idx


def window_overlap_purge(windows: list[WindowSample], train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    """Drop train windows that share frames with any test window."""
    test_ranges = [(windows[int(i)].start, windows[int(i)].end) for i in test_idx]
    keep = []
    for i in train_idx:
        window = windows[int(i)]
        if all(window.end <= start or window.start >= end for start, end in test_ranges):
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def make_split(windows: list[WindowSample], y: np.ndarray, test_fraction: float, seed: int, strategy: str, purge_overlap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "chronological":
        train_idx, test_idx = chronological_split(y, test_fraction)
    elif strategy == "blocked-instance":
        train_idx, test_idx = blocked_instance_split(windows, y, test_fraction)
    elif strategy == "grouped-segment":
        train_idx, test_idx = grouped_segment_split(windows, y, test_fraction, seed)
    elif strategy == "stratified":
        return stratified_split(y, test_fraction, seed)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    if purge_overlap:
        train_idx = window_overlap_purge(windows, train_idx, test_idx)
    return train_idx, test_idx


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


def train_gated_fusion(
    X_rgb: np.ndarray,
    X_hand: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Learned per-window gate between an RGB expert and a hand expert.

    Each modality has its own linear classifier; a gate network reads both
    feature blocks and emits a scalar weight g per window, so the combined
    logits are g * rgb_logits + (1 - g) * hand_logits. Unlike fixed early or
    late fusion, the model can down-weight the weaker modality window by window.

    A naive gate overfits to one expert on training data and collapses (it will
    drive g to 0 or 1 and generalize worse than either modality alone). Two
    standard mixture-of-experts regularizers keep it honest: auxiliary direct
    supervision of each expert, and an entropy bonus that discourages saturated
    gates. The history records the mean RGB gate so you can see what it learned.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for gated fusion. Install with: pip install -e '.[mlp]'") from exc

    Xr, _, _ = fit_scaler(X_rgb, train_idx)
    Xh, _, _ = fit_scaler(X_hand, train_idx)
    torch.manual_seed(seed)
    rgb_tr = torch.as_tensor(Xr[train_idx], dtype=torch.float32)
    hand_tr = torch.as_tensor(Xh[train_idx], dtype=torch.float32)
    y_tr = torch.as_tensor(y[train_idx], dtype=torch.long)
    rgb_te = torch.as_tensor(Xr[test_idx], dtype=torch.float32)
    hand_te = torch.as_tensor(Xh[test_idx], dtype=torch.float32)
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
    weights = len(train_idx) / np.maximum(counts, 1.0) / max(n_classes, 1)

    rgb_expert = torch.nn.Linear(Xr.shape[1], n_classes)
    hand_expert = torch.nn.Linear(Xh.shape[1], n_classes)
    gate = torch.nn.Sequential(
        torch.nn.Linear(Xr.shape[1] + Xh.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.15),
        torch.nn.Linear(64, 1),
    )
    params = list(rgb_expert.parameters()) + list(hand_expert.parameters()) + list(gate.parameters())
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32))
    # The gate saturates and collapses under the softmax learning rate (0.15);
    # full-batch AdamW needs a much smaller step to keep the gate soft.
    gate_lr = min(lr, 5e-3)
    optimizer = torch.optim.AdamW(params, lr=gate_lr, weight_decay=l2)

    def forward(rgb, hand):
        rgb_logits = rgb_expert(rgb)
        hand_logits = hand_expert(hand)
        g = torch.sigmoid(gate(torch.cat([rgb, hand], dim=1)))
        logits = g * rgb_logits + (1.0 - g) * hand_logits
        return logits, g, rgb_logits, hand_logits

    aux_weight, entropy_weight = 0.3, 0.05
    def set_mode(training: bool):
        for module in (rgb_expert, hand_expert, gate):
            module.train(training)

    history = []
    for epoch in range(1, epochs + 1):
        set_mode(True)
        optimizer.zero_grad()
        logits, g, rgb_logits, hand_logits = forward(rgb_tr, hand_tr)
        gate_entropy = -(g * torch.log(g + 1e-6) + (1 - g) * torch.log(1 - g + 1e-6)).mean()
        loss = (
            loss_fn(logits, y_tr)
            + aux_weight * (loss_fn(rgb_logits, y_tr) + loss_fn(hand_logits, y_tr))
            - entropy_weight * gate_entropy
        )
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == epochs:
            pred = logits.detach().argmax(dim=1)
            history.append({
                "epoch": epoch,
                "train_accuracy": float((pred == y_tr).float().mean().item()),
                "train_loss": float(loss.detach().item()),
                "mean_rgb_gate": float(g.detach().mean().item()),
            })
    set_mode(False)
    with torch.no_grad():
        logits, g, _, _ = forward(rgb_te, hand_te)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    history.append({"test_mean_rgb_gate": float(g.detach().mean().item())})
    return probs.astype(np.float32), history


def predict_majority(y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, n_classes: int) -> tuple[np.ndarray, list[dict]]:
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
    majority = int(np.argmax(counts))
    confidence = float(counts[majority] / max(float(counts.sum()), 1.0))
    probs = np.zeros((len(test_idx), n_classes), dtype=np.float32)
    if len(test_idx):
        probs[:, majority] = confidence
    history = [{"epoch": 0, "train_accuracy": float(np.mean(y[train_idx] == majority)), "majority_class_id": majority}]
    return probs, history


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


def top_confusions(cm: np.ndarray, class_names: list[str], k: int = 8) -> list[dict]:
    """Most frequent off-diagonal (true -> predicted) confusions, with directionality.

    Surfaces which action classes the model mixes up; on a single-kitchen episode
    these are usually verbs that share an object (the kettle verbs here).
    """
    rows = []
    for i in range(len(class_names)):
        support = int(cm[i].sum())
        for j in range(len(class_names)):
            if i == j or cm[i, j] == 0:
                continue
            rows.append({
                "true": class_names[i],
                "predicted": class_names[j],
                "count": int(cm[i, j]),
                "fraction_of_true": round(cm[i, j] / support, 4) if support else 0.0,
            })
    rows.sort(key=lambda row: (row["count"], row["fraction_of_true"]), reverse=True)
    return rows[:k]


def calibration_report(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability bins and expected calibration error for argmax confidence."""
    if len(y_true) == 0:
        return {"num_bins": n_bins, "ece": 0.0, "bins": []}
    pred = probs.argmax(axis=1)
    confidence = probs[np.arange(len(pred)), pred]
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        low, high = float(edges[i]), float(edges[i + 1])
        mask = (confidence >= low) & (confidence < high) if i < n_bins - 1 else (confidence >= low) & (confidence <= high)
        count = int(mask.sum())
        if count:
            mean_conf = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += (count / len(y_true)) * abs(accuracy - mean_conf)
        else:
            mean_conf, accuracy = 0.0, 0.0
        bins.append({"bin_low": low, "bin_high": high, "count": count, "mean_confidence": mean_conf, "accuracy": accuracy})
    return {"num_bins": n_bins, "ece": float(ece), "bins": bins}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_predictions(
    name: str,
    probs: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
    class_names: list[str],
    windows: list[WindowSample],
    out_dir: Path | None,
    *,
    model_label: str,
    split_strategy: str,
    history: list[dict] | None = None,
    extra: dict | None = None,
) -> dict:
    pred = probs.argmax(axis=1)
    y_true = y[test_idx]
    metrics, per_class, cm = compute_metrics(y_true, pred, class_names)
    calibration = calibration_report(probs, y_true)
    confusions = top_confusions(cm, class_names)
    metrics["ece"] = calibration["ece"]
    metrics["top_confusions"] = confusions[:3]
    metrics.update({"experiment": name, "model": model_label, "split_strategy": split_strategy})
    if extra:
        metrics.update(extra)
    if out_dir is None:
        return metrics
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if history is not None:
        (exp_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (exp_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
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
    if confusions:
        write_csv(exp_dir / "top_confusions.csv", confusions, ["true", "predicted", "count", "fraction_of_true"])
    return metrics


def run_experiment(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    windows: list[WindowSample],
    out_dir: Path | None,
    args,
    model_type: str = "softmax",
    seed: int | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    seed = args.seed if seed is None else seed
    purge_overlap = getattr(args, "purge_overlap", True)
    train_idx, test_idx = make_split(windows, y, args.test_fraction, seed, args.split_strategy, purge_overlap)
    Xs, mean, std = fit_scaler(X, train_idx)
    if model_type == "majority":
        probs, history = predict_majority(y, train_idx, test_idx, len(class_names))
        model_payload = {"mean": mean, "std": std, "majority": np.asarray([int(np.argmax(np.bincount(y[train_idx], minlength=len(class_names))))], dtype=np.int64)}
    elif model_type == "softmax":
        W, b, history = train_softmax(Xs, y, train_idx, len(class_names), args.epochs, args.learning_rate, args.l2, seed)
        probs = softmax(Xs[test_idx] @ W + b)
        model_payload = {"mean": mean, "std": std, "W": W, "b": b}
    elif model_type == "mlp":
        probs, history, model = train_torch_mlp(Xs, y, train_idx, test_idx, len(class_names), args.epochs, args.learning_rate, args.l2, seed, args.mlp_hidden_dim)
        model_payload = {"mean": mean, "std": std, "torch_model": model}
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    metrics = evaluate_predictions(
        name,
        probs,
        y,
        test_idx,
        class_names,
        windows,
        out_dir,
        model_label=model_type,
        split_strategy=args.split_strategy,
        history=history,
        extra={
            "feature_dim": int(X.shape[1]),
            "num_windows": int(len(y)),
            "num_train": int(len(train_idx)),
            "num_test": int(len(test_idx)),
            "purge_overlap": bool(purge_overlap),
            "seed": int(seed),
        },
    )
    if out_dir is not None:
        exp_dir = out_dir / name
        if model_type in ("softmax", "majority"):
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
    return metrics, probs, test_idx


def build_dataset(data_root: Path, video_name: str, target: str, window_frames: int, stride_frames: int, min_label_fraction: float, max_windows: int | None, rgb_embedding: str = "handcrafted"):
    annotation = data_root / "annotation.hdf5"
    caption = load_caption(annotation)
    frame_numbers = load_frame_numbers(annotation)
    labels = frame_labels_from_caption(caption, frame_numbers, target)
    windows = build_windows(labels, window_frames, stride_frames, min_label_fraction, max_windows)
    y, class_names = encode_labels([w.label for w in windows])
    left, right = load_hand_joints(annotation)
    X_hand = hand_features(left, right, windows)
    frame_feature_fn = make_dino_embedder() if rgb_embedding == "dino" else None
    X_rgb = rgb_features(data_root / video_name, windows, frame_feature_fn)
    return windows, y, class_names, X_rgb, X_hand
