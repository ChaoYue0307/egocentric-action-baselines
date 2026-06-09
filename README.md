# Egocentric Action Baselines

Learn how first-person action recognition works by comparing three small,
inspectable baselines on one Xperience-10M pour-over coffee episode.

The task is simple to state: given a short temporal window from an egocentric
video, predict what action is happening. The repo shows how RGB appearance,
hand-joint motion, and early sensor fusion each contribute to that prediction.

## Interactive Tutorial

Open the visual walkthrough:

- Web page: https://chaoyue0307.github.io/egocentric-action-baselines/
- Local copy: open `docs/index.html` in a browser.

The page explains the pipeline, key metrics, and ablation result with an
interactive chart. A deeper glossary lives in `docs/concepts.md`.

## What You Will Learn

- **Egocentric action recognition:** classifying actions from the camera wearer's point of view.
- **Temporal windows:** grouping nearby frames so the model sees motion, not only one image.
- **RGB baseline:** using image color, texture, and coarse layout features.
- **Hand-joint baseline:** using 3D hand pose and motion as an action cue.
- **Fusion baseline:** concatenating RGB and hand features before classification.
- **Ablation study:** changing one input source at a time to see what matters.
- **Accuracy and F1:** reading both overall correctness and class-balanced performance.

## Data

Raw Xperience-10M videos, `annotation.hdf5`, and `.rrd` files stay outside this repository.
Set `DATA_ROOT` to your local sample directory:

```bash
export DATA_ROOT=/path/to/xperience-10m-sample
```

The expected directory contains `annotation.hdf5` and `fisheye_cam0.mp4`.
See `DATA_NOTICE.md` for the minimal data contract.

## Run The Baselines

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_ablation.py \
  --data-root "$DATA_ROOT" \
  --output-dir outputs/sample_ablation \
  --max-windows 240
```

Use `--max-windows 0` to run all labeled windows.

## Baselines

| Experiment | Signal | What It Tests |
| --- | --- | --- |
| `rgb_only` | sampled frames from `fisheye_cam0.mp4` | Can appearance and scene layout identify the action? |
| `hand_joints_only` | left/right 3D hand joints from `annotation.hdf5` | Is hand motion enough to infer intent? |
| `rgb_hand_fusion` | RGB + hand features | Does combining visual context with hand motion help? |

Sample result from a short run:

| Experiment | Accuracy | Macro F1 | Feature Dim |
| --- | ---: | ---: | ---: |
| `rgb_only` | 1.000 | 1.000 | 686 |
| `hand_joints_only` | 0.933 | 0.907 | 882 |
| `rgb_hand_fusion` | 0.933 | 0.907 | 1568 |

Each run writes `summary.json` plus per-experiment `metrics.json`,
`per_class_metrics.csv`, `confusion_matrix.csv`, `predictions.csv`, and
`model.npz`.

## How To Read The Result

High scores on one episode mean the feature extraction, label alignment, and
evaluation loop are working. They do not prove cross-episode generalization.
For a stronger benchmark, add many episodes and split by held-out episode so the
test set contains unseen kitchens, camera motion, and action styles.
