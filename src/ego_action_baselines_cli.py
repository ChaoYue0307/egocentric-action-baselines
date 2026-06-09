from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[2] / "data/sample/xperience-10m-sample"
    parser = argparse.ArgumentParser(description="Run RGB, hand-joints, and fusion action baselines on one egocentric Xperience sample.")
    parser.add_argument("--data-root", type=Path, default=root_default)
    parser.add_argument("--data-roots", type=Path, nargs="*", help="Optional list of episode roots. When set, each episode is evaluated and aggregate metrics are written.")
    parser.add_argument("--video", default="fisheye_cam0.mp4")
    parser.add_argument("--target", choices=["action", "subtask"], default="action")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--min-label-fraction", type=float, default=0.6)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-strategy", choices=["chronological", "stratified"], default="chronological", help="Use chronological to reduce overlap leakage; use stratified for quick class-balanced checks.")
    parser.add_argument("--max-windows", type=int, default=240, help="Limit windows for quick local runs. Use 0 for all windows.")
    parser.add_argument("--model", choices=["majority", "softmax", "mlp", "both", "all"], default="softmax", help="Classifier head to run. MLP requires PyTorch.")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128, help="Hidden width for the optional PyTorch MLP baseline.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def model_types_for(name: str) -> list[str]:
    if name == "both":
        return ["softmax", "mlp"]
    if name == "all":
        return ["majority", "softmax", "mlp"]
    return [name]


def summarize_episode(args, data_root: Path, output_dir: Path) -> dict:
    import numpy as np

    from ego_action_baselines import build_dataset, run_experiment

    max_windows = None if args.max_windows == 0 else args.max_windows
    output_dir.mkdir(parents=True, exist_ok=True)
    windows, y, class_names, X_rgb, X_hand = build_dataset(
        data_root,
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
        "data_root": str(data_root),
        "video": args.video,
        "target": args.target,
        "num_windows": len(windows),
        "split_strategy": args.split_strategy,
        "model": args.model,
        "classes": class_names,
        "experiments": {},
    }
    model_types = model_types_for(args.model)
    for name, X in datasets.items():
        for model_type in model_types:
            exp_name = name if model_type == "softmax" else f"{name}_{model_type}"
            metrics = run_experiment(exp_name, X, y, class_names, windows, output_dir, args, model_type)
            summary["experiments"][exp_name] = metrics
            print(f"{exp_name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def aggregate_summaries(summaries: list[dict]) -> dict:
    import numpy as np

    experiment_names = sorted({name for summary in summaries for name in summary["experiments"]})
    aggregate = {"num_episodes": len(summaries), "experiments": {}}
    for name in experiment_names:
        rows = [summary["experiments"][name] for summary in summaries if name in summary["experiments"]]
        aggregate["experiments"][name] = {
            "episodes": len(rows),
            "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
            "accuracy_std": float(np.std([row["accuracy"] for row in rows])),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in rows])),
        }
    return aggregate


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_roots = args.data_roots if args.data_roots else [args.data_root]
    summaries = []
    for idx, data_root in enumerate(data_roots):
        episode_dir = args.output_dir if len(data_roots) == 1 else args.output_dir / f"episode_{idx:02d}"
        print(f"episode={idx} data_root={data_root}")
        summaries.append(summarize_episode(args, data_root, episode_dir))
    if len(summaries) > 1:
        aggregate = aggregate_summaries(summaries)
        (args.output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return 0
