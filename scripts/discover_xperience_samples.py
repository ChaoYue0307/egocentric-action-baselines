#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_episode(path: Path, video_name: str) -> bool:
    return (path / "annotation.hdf5").exists() and (path / video_name).exists()


def discover(search_roots: list[Path], video_name: str) -> list[dict]:
    episodes = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        candidates = [root] + [path.parent for path in root.rglob("annotation.hdf5")]
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen or not is_episode(path, video_name):
                continue
            seen.add(resolved)
            episodes.append({
                "path": str(path),
                "annotation": str(path / "annotation.hdf5"),
                "video": str(path / video_name),
            })
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover local Xperience-style episode folders for multi-episode evaluation.")
    parser.add_argument("--search-root", type=Path, action="append", default=[Path("../data")])
    parser.add_argument("--video", default="fisheye_cam0.mp4")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/sample_ablation/episode_discovery.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episodes = discover(args.search_root, args.video)
    payload = {
        "num_episodes": len(episodes),
        "episodes": episodes,
        "note": "The public Xperience-10M sample distribution currently exposes one sample episode; full multi-episode data is gated.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
