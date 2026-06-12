# Action Baseline Report

## Motivation

Egocentric action recognition asks what the camera wearer is doing from a short
first-person temporal window. The project compares appearance, hand motion, and
two fusion strategies on one pour-over coffee episode, and uses the split
design itself as a teaching instrument.

## Method

The pipeline builds labeled temporal windows from `annotation.hdf5`, extracts
RGB summary features from sampled frames (or optional frozen DINOv2
embeddings), extracts temporal statistics from 3D hand joints, and trains a
NumPy softmax head or an optional PyTorch MLP head. Late fusion averages the
predicted probabilities of the RGB-only and hand-only models.

Four split strategies are implemented:

- `stratified`: random class-balanced windows — leaky upper bound, since
  adjacent windows share up to 15 of 20 frames.
- `chronological`: timeline tail held out — leak-free but label-shifted,
  because every action in this episode occurs exactly once.
- `blocked-instance` (default): tail of each action instance held out, plus
  purging of train windows that share frames with test windows.
- `grouped-segment`: whole action instances held out — the honest
  cross-instance protocol, which refuses to run on single-instance episodes.

## Results

Full episode, 1093 windows, 18 classes, mean ± std over 5 seeds
(`outputs/sample_ablation/summary.json`):

| Experiment | Accuracy | Macro F1 |
| --- | ---: | ---: |
| majority | 0.143 ± 0.000 | 0.015 ± 0.000 |
| rgb_only | 0.254 ± 0.002 | 0.163 ± 0.003 |
| hand_joints_only | 0.469 ± 0.002 | 0.267 ± 0.007 |
| rgb_hand_fusion (early) | 0.358 ± 0.002 | 0.264 ± 0.005 |
| rgb_hand_late_fusion | 0.335 ± 0.005 | 0.207 ± 0.015 |

Cross-split comparison for `hand_joints_only`
(`outputs/split_comparison/`): stratified 0.941, chronological 0.004,
blocked-instance 0.471. The spread is the headline result: the same model and
features span two orders of magnitude depending on evaluation design.

## Artifacts

- `summary.json`: experiment-level metrics plus seed variance.
- `metrics.json`: accuracy, macro F1, weighted F1, balanced accuracy, ECE.
- `calibration.json`: reliability bins and expected calibration error.
- `per_class_metrics.csv`: class-level precision, recall, and F1.
- `predictions.csv`: window-level predictions and confidence.
- `episode_discovery.json`: local episode discovery for batch evaluation.

## Interpretation

Hand-joint motion is the strongest single cue on this episode; handcrafted RGB
summaries mostly encode scene layout, which barely changes within one kitchen.
Early fusion lands between its inputs rather than above them — the weaker RGB
half competes with hand features inside one linear head. Late fusion is better
calibrated but averages away correct hand predictions. All learned baselines
are heavily overconfident (ECE 0.30–0.47).

Scores on one episode validate the data path and evaluation mechanics and
measure within-instance generalization only. They should not be read as
cross-episode generalization. A stronger benchmark should split by held-out
episodes and include more tasks, homes, camera motions, and object
arrangements.

The current public Xperience-10M sample distribution exposes one verified
sample episode. The CLI supports `--data-roots` for multi-episode runs, and
`scripts/discover_xperience_samples.py` records which local episodes are
available before running a batch evaluation.

## Failure Modes

- Stratified splits leak shared frames through overlapping windows.
- Chronological tail splits suffer total label shift on single-instance episodes.
- RGB features can overfit scene layout.
- Hand joints can be noisy or missing.
- Early fusion lets a weak modality drag down a strong one.
- MLP results can vary more with seed and small sample size.

## Next Work

- Held-out episode evaluation when more episodes are available
  (`grouped-segment` is already implemented and waiting for that data).
- Run the frozen DINOv2 embedding baseline (`--rgb-embedding dino`) and compare
  representation quality under the identical head and split.
- Compare lightweight temporal models such as 1D CNNs or GRUs.
- Attention-style or gated fusion that can ignore the weaker modality per
  window instead of averaging it in.
