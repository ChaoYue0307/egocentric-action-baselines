#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def bar(x: int, y: int, width: int, label: str, value: float, color: str) -> str:
    filled = int(width * value)
    return f"""
    <text x="{x}" y="{y - 8}" fill="#cbd5e1" font-size="15" font-family="Inter, Arial">{label}</text>
    <rect x="{x}" y="{y}" width="{width}" height="18" rx="9" fill="#0f172a" stroke="#334155"/>
    <rect x="{x}" y="{y}" width="{filled}" height="18" rx="9" fill="{color}"/>
    <text x="{x + width + 16}" y="{y + 15}" fill="#e2e8f0" font-size="15" font-family="Inter, Arial">{value:.3f}</text>
    """


def render(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    experiments = summary["experiments"]
    rows = [
        ("RGB only", experiments["rgb_only"], "#38bdf8"),
        ("Hand joints only", experiments["hand_joints_only"], "#a78bfa"),
        ("RGB + hand fusion", experiments["rgb_hand_fusion"], "#34d399"),
    ]
    parts = []
    y = 104
    for label, metrics, color in rows:
        parts.append(bar(78, y, 360, label, float(metrics["macro_f1"]), color))
        y += 66
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360">
  <rect width="760" height="360" rx="28" fill="#020617"/>
  <text x="48" y="48" fill="#f8fafc" font-size="26" font-weight="700" font-family="Inter, Arial">Action Baseline Macro F1</text>
  <text x="48" y="76" fill="#94a3b8" font-size="15" font-family="Inter, Arial">Sample: {summary['num_windows']} windows · split: {summary.get('split_strategy', 'stratified')} · target: {summary['target']}</text>
  {''.join(parts)}
  <text x="48" y="326" fill="#64748b" font-size="13" font-family="Inter, Arial">Macro F1 weights each action class equally.</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> int:
    render(Path("outputs/sample_ablation/summary.json"), Path("docs/assets/action_metrics.svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
