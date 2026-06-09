from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    parser.add_argument("--model", choices=["softmax", "mlp", "both"], default="softmax", help="Classifier head to run. MLP requires PyTorch.")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128, help="Hidden width for the optional PyTorch MLP baseline.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import numpy as np

    from ego_action_baselines import build_dataset, run_experiment

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
        "model": args.model,
        "classes": class_names,
        "experiments": {},
    }
    model_types = ["softmax", "mlp"] if args.model == "both" else [args.model]
    for name, X in datasets.items():
        for model_type in model_types:
            exp_name = name if model_type == "softmax" else f"{name}_mlp"
            metrics = run_experiment(exp_name, X, y, class_names, windows, args.output_dir, args, model_type)
            summary["experiments"][exp_name] = metrics
            print(f"{exp_name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0
