# Egocentric Action Baselines

[![CI](https://github.com/ChaoYue0307/egocentric-action-baselines/actions/workflows/ci.yml/badge.svg)](https://github.com/ChaoYue0307/egocentric-action-baselines/actions/workflows/ci.yml)

Learn how first-person action recognition works by comparing three small,
inspectable baselines on one Xperience-10M pour-over coffee episode.

Part of the Egocentric Vision Learning Hub:
https://chaoyue0307.github.io/egocentric-vision-learning-hub/

The task is simple to state: given a short temporal window from an egocentric
video, predict what action is happening. The repo shows how RGB appearance,
hand-joint motion, and early sensor fusion each contribute to that prediction.

![Egocentric action baseline tutorial preview](docs/assets/readme_preview.svg)
![Animated action demo loop](docs/assets/demo_loop.svg)
![Live tutorial screenshot](docs/assets/live_screenshot.png)

Short walkthrough recording: [`docs/assets/walkthrough.webm`](docs/assets/walkthrough.webm)

## Interactive Tutorial

Open the visual walkthrough:

- Web page: https://chaoyue0307.github.io/egocentric-action-baselines/
- Local copy: open `docs/index.html` in a browser.

The page includes a learning path, metric selector, ablation chart, visual
concept explanations, and result interpretation cards.

## What You Will Learn

- **Egocentric action recognition:** classifying actions from the camera wearer's point of view.
- **Temporal windows:** grouping nearby frames so the model sees motion, not only one image.
- **RGB baseline:** using image color, texture, and coarse layout features.
- **Hand-joint baseline:** using 3D hand pose and motion as an action cue.
- **Early vs late fusion:** concatenating features before training versus averaging predictions after.
- **Ablation study:** changing one input source at a time to see what matters.
- **Accuracy and F1:** reading both overall correctness and class-balanced performance.
- **Split design:** why the train/test split decides what your numbers mean.
- **Leakage and label shift:** the two classic ways a single-episode evaluation lies to you.
- **Calibration:** whether predicted confidence matches empirical accuracy (ECE).

## Data

Raw Xperience-10M videos, `annotation.hdf5`, and `.rrd` files stay outside this repository.
Set `DATA_ROOT` to your local sample directory:

```bash
export DATA_ROOT=/path/to/xperience-10m-sample
```

The expected directory contains `annotation.hdf5` and `fisheye_cam0.mp4`.
See `DATA_NOTICE.md`, `DATA_CARD.md`, and `EVALUATION_CARD.md` for the data contract, intended use, metrics, and limitations.

## Run The Baselines

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_ablation.py \
  --data-root "$DATA_ROOT" \
  --output-dir outputs/sample_ablation \
  --model classical \
  --seeds 5 \
  --max-windows 0
```

Use `--max-windows 0` to run all labeled windows (recommended: short prefixes
of the episode contain each action only once, which starves some splits).
The default `blocked-instance` split holds out the chronological tail of each
action instance and purges train windows that share frames with test windows.

After installing the project, the same command is available as:

```bash
pip install -e .
ego-action-ablation --data-root "$DATA_ROOT" --output-dir outputs/sample_ablation
```

To compare the NumPy softmax head with the optional PyTorch MLP head:

```bash
pip install -e ".[mlp]"
ego-action-ablation --data-root "$DATA_ROOT" --model both --epochs 300
```

To run a stronger multi-episode check when you have more samples:

```bash
python scripts/discover_xperience_samples.py --search-root ../data
ego-action-ablation \
  --data-roots /path/to/sample_a /path/to/sample_b \
  --model all \
  --output-dir outputs/multi_episode_eval
```

## Repository Map

| Path | Purpose |
| --- | --- |
| `scripts/run_ablation.py` | command-line entry point for RGB, hand, and fusion experiments |
| `src/ego_action_baselines.py` | dataset loading, feature extraction, training, and metrics |
| `src/adapters.py` | source boundary for video frames, hand poses, and labels |
| `notebooks/01_action_baseline_walkthrough.ipynb` | step-by-step notebook companion |
| `reports/action_baseline_report.md` | paper-style method, result, and limitation summary |
| `docs/index.html` | interactive tutorial webpage |
| `docs/concepts.md` | glossary for CV/ML terms |
| `outputs/sample_ablation/summary.json` | compact example result |

## Common Commands

```bash
make test
make help
make visuals
make pages
```

## Baselines

| Experiment | Signal | What It Tests |
| --- | --- | --- |
| `rgb_only` | sampled frames from `fisheye_cam0.mp4` | Can appearance and scene layout identify the action? |
| `hand_joints_only` | left/right 3D hand joints from `annotation.hdf5` | Is hand motion enough to infer intent? |
| `rgb_hand_fusion` | RGB + hand features concatenated (early fusion) | Does combining features before training help? |
| `rgb_hand_late_fusion` | average of RGB and hand prediction probabilities | Does combining predictions after training help more? |
| `rgb_hand_gated_fusion` | learned per-window gate between RGB and hand experts (`--gated-fusion`) | Can the model learn *when* to trust each modality? |
| `*_majority` | no visual or hand features | Does the classifier beat the class-prior baseline? |
| `*_mlp` | same features with a small PyTorch MLP head | Does a nonlinear classifier change the ranking? |

Result on the full sample episode (1093 windows, 18 action classes,
`blocked-instance` split, mean ± std over 5 seeds):

| Experiment | Accuracy | Macro F1 | ECE |
| --- | ---: | ---: | ---: |
| `*_majority` | 0.143 ± 0.000 | 0.015 ± 0.000 | 0.005 |
| `rgb_only` | 0.254 ± 0.002 | 0.163 ± 0.003 | 0.469 |
| `hand_joints_only` | **0.469 ± 0.002** | 0.267 ± 0.007 | 0.357 |
| `rgb_hand_fusion` (early) | 0.358 ± 0.002 | **0.264 ± 0.005** | 0.464 |
| `rgb_hand_late_fusion` | 0.335 ± 0.005 | 0.207 ± 0.015 | 0.302 |
| `rgb_hand_gated_fusion` | 0.355 ± 0.006 | 0.262 ± 0.010 | — |

The gated fusion run requires PyTorch (`--gated-fusion`).

Each run writes `summary.json` plus per-experiment `metrics.json`,
`per_class_metrics.csv`, `confusion_matrix.csv`, `predictions.csv`,
`calibration.json` (reliability bins and ECE), and `model.npz`.
With `--seeds N` the summary adds mean and standard deviation across seeds.
Batch runs also write `aggregate_summary.json` across episode roots.

## The Split Decides What The Numbers Mean

The same features and models produce wildly different numbers depending on the
split, and each gap teaches a different evaluation failure
(`outputs/split_comparison/` holds the committed summaries):

| Split | `hand_joints_only` accuracy | What it measures |
| --- | ---: | --- |
| `stratified` | 0.941 | **Leakage.** Random windows overlap by up to 15 of 20 frames across train/test, so the model partly memorizes shared frames. A leaky upper bound. |
| `chronological` | 0.004 | **Label shift.** Every action in this episode happens exactly once, so the timeline tail contains classes the model never trained on. Even the majority baseline scores 0.0. |
| `blocked-instance` | 0.471 | **Within-instance generalization.** The tail of each action instance is held out and overlapping train windows are purged, so every test class has train support and no frame is shared. |
| `grouped-segment` | n/a here | **Cross-instance generalization.** Holds out whole action instances; needs repeated instances or multiple episodes, so it refuses to run on this single-instance episode. |

Three honest readings of the blocked-instance result:

- **Hands beat pixels.** Hand-joint motion is the strongest single cue across
  all 18 actions; handcrafted RGB summaries mostly capture scene layout, which
  changes little within one kitchen.
- **Fusion is not free.** All three fusion strategies land *between* their two
  inputs, never above the better one (hands at 0.469). Early fusion (0.358) lets
  the 882 hand dimensions and 686 RGB dimensions compete inside one linear head.
  Late fusion (0.335) is better calibrated but averages away correct hand
  predictions. **Gated fusion** (0.355), which learns a per-window weight
  between an RGB expert and a hand expert, only recovers early-fusion accuracy —
  and its gate settles near 0.51, i.e. it learns no consistent rule for *when*
  to prefer a modality. (Without regularization the gate collapses to one expert
  and does worse; the implementation uses auxiliary per-expert supervision and a
  small gate learning rate, both documented in `train_gated_fusion`.) On one
  episode there is simply not enough signal for a learned gate to beat fixed
  fusion. Making fusion actually exceed the best single modality is the real
  research exercise this repo hands you.
- **Confidence lies.** ECE around 0.35–0.47 means predicted confidence runs far
  ahead of empirical accuracy. Check `calibration.json` before trusting any
  probability from a small model on a small dataset.
- **Confusions are about objects, not motions.** `top_confusions.csv` shows the
  errors cluster among verbs that share an object: `Hold gooseneck kettle` →
  `Grasp gooseneck kettle` (49% of its windows), `Grasp coffee scoop` →
  `Transfer coffee to dripper` (100%), `Lift gooseneck kettle` → `Move kettle
  away` (87%). The hard part of this task is verb granularity on the same
  object, not telling the kettle from the scoop — a genuinely egocentric
  difficulty that more RGB resolution alone will not fix.

High scores on one episode still do not prove cross-episode generalization.
For a stronger benchmark, add many episodes and split by held-out episode so the
test set contains unseen kitchens, camera motion, and action styles.

## Frozen Pretrained Embeddings (Measured)

Swapping the handcrafted RGB features for frozen DINOv2 embeddings, under the
*identical* softmax head and `blocked-instance` split, isolates the effect of
the representation:

```bash
pip install -e ".[dino]"
ego-action-ablation --data-root "$DATA_ROOT" --rgb-embedding dino --model classical --output-dir outputs/dino_ablation
```

| Experiment | Handcrafted RGB | DINOv2 (frozen) |
| --- | ---: | ---: |
| `rgb_only` | 0.254 | **0.312** |
| `rgb_hand_fusion` (early) | 0.358 | **0.408** |
| `rgb_hand_late_fusion` | 0.335 | **0.393** |

DINOv2 (ViT-S/14, 384-d) lifts RGB-only accuracy by ~6 points and pulls every
fusion variant up with it (committed in `outputs/dino_ablation/summary.json`).
The representation clearly helps — yet hands alone (0.469) still win, so the
headline finding survives a much stronger visual backbone. First run downloads
hub weights, so this never runs in CI.
