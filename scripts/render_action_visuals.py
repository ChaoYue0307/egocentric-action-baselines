#!/usr/bin/env python3
from __future__ import annotations

import json
import csv
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
        ("Majority baseline", experiments.get("rgb_only_majority"), "#64748b"),
        ("RGB only", experiments["rgb_only"], "#38bdf8"),
        ("Hand joints only", experiments["hand_joints_only"], "#a78bfa"),
        ("RGB + hand early fusion", experiments["rgb_hand_fusion"], "#34d399"),
        ("RGB + hand late fusion", experiments.get("rgb_hand_late_fusion"), "#fbbf24"),
    ]
    parts = []
    y = 104
    for label, metrics, color in rows:
        if metrics is None:
            continue
        parts.append(bar(78, y, 360, label, float(metrics["macro_f1"]), color))
        y += 66
    height = y + 60
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" viewBox="0 0 760 {height}">
  <rect width="760" height="{height}" rx="28" fill="#020617"/>
  <text x="48" y="48" fill="#f8fafc" font-size="26" font-weight="700" font-family="Inter, Arial">Action Baseline Macro F1</text>
  <text x="48" y="76" fill="#94a3b8" font-size="15" font-family="Inter, Arial">Sample: {summary['num_windows']} windows · split: {summary.get('split_strategy', 'stratified')} · target: {summary['target']}</text>
  {''.join(parts)}
  <text x="48" y="{height - 24}" fill="#64748b" font-size="13" font-family="Inter, Arial">Macro F1 weights each action class equally.</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def render_confusion_matrix(csv_path: Path, output_path: Path) -> None:
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fp:
            rows = list(csv.reader(fp))
        labels = rows[0][1:]
        matrix = [[int(x) for x in row[1:]] for row in rows[1:]]
        subtitle = f"Generated from {csv_path}"
    else:
        labels = ["action 0", "action 1"]
        matrix = [[21, 0], [9, 0]]
        subtitle = "Preview from the committed sample summary; rerun visuals after a full experiment for exact counts."
    max_value = max([value for row in matrix for value in row] + [1])
    cell = 54
    left = 150
    top = 116
    cells = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            opacity = 0.18 + 0.72 * value / max_value
            cells.append(f'<rect x="{left + j * cell}" y="{top + i * cell}" width="{cell - 4}" height="{cell - 4}" rx="10" fill="#38bdf8" opacity="{opacity:.2f}"/>')
            cells.append(f'<text x="{left + j * cell + 25}" y="{top + i * cell + 31}" text-anchor="middle" fill="#e2e8f0" font-family="Inter, Arial" font-size="14">{value}</text>')
    label_text = []
    for i, _label in enumerate(labels):
        label_text.append(f'<text x="{left + i * cell + 25}" y="102" text-anchor="middle" fill="#94a3b8" font-family="Inter, Arial" font-size="12">{i}</text>')
        label_text.append(f'<text x="132" y="{top + i * cell + 31}" text-anchor="end" fill="#94a3b8" font-family="Inter, Arial" font-size="12">{i}</text>')
    legend = " · ".join(f"{idx}: {label}" for idx, label in enumerate(labels))
    height = max(360, top + len(labels) * cell + 88)
    width = max(760, left + len(labels) * cell + 70)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" rx="28" fill="#020617"/>
  <text x="48" y="48" fill="#f8fafc" font-size="26" font-weight="700" font-family="Inter, Arial">Confusion Matrix</text>
  <text x="48" y="76" fill="#94a3b8" font-size="13" font-family="Inter, Arial">{subtitle}</text>
  <text x="48" y="{height - 44}" fill="#94a3b8" font-size="12" font-family="Inter, Arial">{legend}</text>
  <text x="{left}" y="{height - 18}" fill="#64748b" font-size="13" font-family="Inter, Arial">columns: predicted · rows: true</text>
  {''.join(label_text)}
  {''.join(cells)}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> int:
    render(Path("outputs/sample_ablation/summary.json"), Path("docs/assets/action_metrics.svg"))
    render_confusion_matrix(Path("outputs/sample_ablation/hand_joints_only/confusion_matrix.csv"), Path("docs/assets/confusion_matrix.svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
