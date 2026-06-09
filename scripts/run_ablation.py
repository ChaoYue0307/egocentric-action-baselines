#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ego_action_baselines import build_dataset, run_experiment


def parse_args() -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[2] / "data/sample/xperience-10m-sample"
    parser = argparse.ArgumentParser(description="Run RGB, hand-joints, and fusion action baselines on one egocentric Xperience sample.")
    parser.add_argument("--data-root", type=Path, default=root_default)
    parser.add_argument("--video", default="fisheye_cam0.mp4")
    parser.add_argument("--target", choices=["action", "subtask"], default="action")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--min-label-fraction", type=float, default=0.6)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-strategy", choices=["chronological", "stratified"], default="chronological", help="Use chronological to reduce overlap leakage; use stratified for quick class-balanced checks.")
    parser.add_argument("--max-windows", type=int, default=240, help="Limit windows for quick local runs. Use 0 for all windows.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_windows = None if args.max_windows == 0 else args.max_windows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows, y, class_names, X_rgb, X_hand = build_dataset(
        args.data_root,
        args.video,
        args.target,
        args.window_frames,
        args.stride_frames,
        args.min_label_fraction,
        max_windows,
    )
    datasets = {
        "rgb_only": X_rgb,
        "hand_joints_only": X_hand,
        "rgb_hand_fusion": np.concatenate([X_rgb, X_hand], axis=1).astype(np.float32),
    }
    summary = {
        "data_root": str(args.data_root),
        "video": args.video,
        "target": args.target,
        "num_windows": len(windows),
        "split_strategy": args.split_strategy,
        "classes": class_names,
        "experiments": {},
    }
    for name, X in datasets.items():
        metrics = run_experiment(name, X, y, class_names, windows, args.output_dir, args)
        summary["experiments"][name] = metrics
        print(f"{name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
