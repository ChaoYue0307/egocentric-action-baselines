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
    parser.add_argument("--split-strategy", choices=["blocked-instance", "chronological", "grouped-segment", "stratified"], default="blocked-instance", help="blocked-instance holds out the tail of each action instance (within-episode); grouped-segment holds out whole instances (needs repeated instances or multiple episodes); chronological holds out the timeline tail; stratified is a leaky upper-bound check.")
    parser.add_argument("--purge-overlap", action=argparse.BooleanOptionalAction, default=True, help="Drop train windows that share frames with any test window (chronological and grouped-segment only).")
    parser.add_argument("--max-windows", type=int, default=240, help="Limit windows for quick local runs. Use 0 for all windows.")
    parser.add_argument("--model", choices=["majority", "softmax", "mlp", "classical", "both", "all"], default="softmax", help="Classifier head to run. MLP requires PyTorch.")
    parser.add_argument("--rgb-embedding", choices=["handcrafted", "dino"], default="handcrafted", help="RGB frame features: handcrafted color/edge stats or frozen DINOv2 embeddings (requires PyTorch).")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128, help="Hidden width for the optional PyTorch MLP baseline.")
    parser.add_argument("--gated-fusion", action="store_true", help="Also train a learned per-window gated fusion of RGB and hand experts (requires PyTorch).")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=1, help="Run N seeds starting at --seed and report mean and std across seeds.")
    return parser.parse_args()


def model_types_for(name: str) -> list[str]:
    if name == "both":
        return ["softmax", "mlp"]
    if name == "classical":
        return ["majority", "softmax"]
    if name == "all":
        return ["majority", "softmax", "mlp"]
    return [name]


def experiment_name(base: str, model_type: str) -> str:
    return base if model_type == "softmax" else f"{base}_{model_type}"


def run_seed_pass(args, datasets, y, class_names, windows, output_dir: Path | None, seed: int) -> dict[str, dict]:
    from ego_action_baselines import evaluate_predictions, make_split, run_experiment, train_gated_fusion

    results: dict[str, dict] = {}
    for name, X in datasets.items():
        for model_type in model_types_for(args.model):
            exp_name = experiment_name(name, model_type)
            metrics, probs, test_idx = run_experiment(exp_name, X, y, class_names, windows, output_dir, args, model_type, seed=seed)
            results[exp_name] = {"metrics": metrics, "probs": probs, "test_idx": test_idx}
    for model_type in model_types_for(args.model):
        if model_type == "majority":
            continue
        rgb_name = experiment_name("rgb_only", model_type)
        hand_name = experiment_name("hand_joints_only", model_type)
        if rgb_name not in results or hand_name not in results:
            continue
        fusion_name = experiment_name("rgb_hand_late_fusion", model_type)
        probs = (results[rgb_name]["probs"] + results[hand_name]["probs"]) / 2.0
        test_idx = results[rgb_name]["test_idx"]
        metrics = evaluate_predictions(
            fusion_name,
            probs,
            y,
            test_idx,
            class_names,
            windows,
            output_dir,
            model_label=f"late_fusion_{model_type}",
            split_strategy=args.split_strategy,
            history=[{"note": f"average of {rgb_name} and {hand_name} probabilities"}],
            extra={"num_test": int(len(test_idx)), "purge_overlap": bool(args.purge_overlap), "seed": int(seed)},
        )
        results[fusion_name] = {"metrics": metrics, "probs": probs, "test_idx": test_idx}
    if getattr(args, "gated_fusion", False) and "rgb_only" in datasets and "hand_joints_only" in datasets:
        train_idx, test_idx = make_split(windows, y, args.test_fraction, seed, args.split_strategy, args.purge_overlap)
        probs, history = train_gated_fusion(
            datasets["rgb_only"], datasets["hand_joints_only"], y, train_idx, test_idx,
            len(class_names), args.epochs, args.learning_rate, args.l2, seed,
        )
        metrics = evaluate_predictions(
            "rgb_hand_gated_fusion", probs, y, test_idx, class_names, windows, output_dir,
            model_label="gated_fusion", split_strategy=args.split_strategy, history=history,
            extra={"num_test": int(len(test_idx)), "purge_overlap": bool(args.purge_overlap), "seed": int(seed),
                   "mean_rgb_gate": history[-1].get("test_mean_rgb_gate")},
        )
        results["rgb_hand_gated_fusion"] = {"metrics": metrics, "probs": probs, "test_idx": test_idx}
    return results


def seed_variance(per_seed: list[dict[str, dict]]) -> dict:
    import numpy as np

    names = sorted({name for results in per_seed for name in results})
    variance = {}
    for name in names:
        rows = [results[name]["metrics"] for results in per_seed if name in results]
        variance[name] = {
            "seeds": len(rows),
            "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
            "accuracy_std": float(np.std([row["accuracy"] for row in rows])),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in rows])),
        }
    return variance


def summarize_episode(args, data_root: Path, output_dir: Path) -> dict:
    import numpy as np

    from ego_action_baselines import build_dataset

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
        rgb_embedding=args.rgb_embedding,
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
        "purge_overlap": bool(args.purge_overlap),
        "rgb_embedding": args.rgb_embedding,
        "model": args.model,
        "classes": class_names,
        "experiments": {},
    }
    n_seeds = max(1, args.seeds)
    per_seed = []
    for k in range(n_seeds):
        seed = args.seed + k
        results = run_seed_pass(args, datasets, y, class_names, windows, output_dir if k == 0 else None, seed)
        per_seed.append(results)
    for exp_name, payload in per_seed[0].items():
        summary["experiments"][exp_name] = payload["metrics"]
        metrics = payload["metrics"]
        print(f"{exp_name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    if n_seeds > 1:
        summary["seed_variance"] = seed_variance(per_seed)
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
