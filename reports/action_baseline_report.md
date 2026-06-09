# Action Baseline Report

## Motivation

Egocentric action recognition asks what the camera wearer is doing from a short
first-person temporal window. The project compares appearance, hand motion, and
early fusion on one pour-over coffee episode.

## Method

The pipeline builds labeled temporal windows from `annotation.hdf5`, extracts
RGB summary features from sampled frames, extracts temporal statistics from 3D
hand joints, and trains either a NumPy softmax head or an optional PyTorch MLP
head. The default chronological split keeps the tail of the episode for
evaluation to reduce overlap leakage.

## Artifacts

- `summary.json`: experiment-level metrics.
- `metrics.json`: accuracy, macro F1, weighted F1, balanced accuracy.
- `per_class_metrics.csv`: class-level precision, recall, and F1.
- `predictions.csv`: window-level predictions and confidence.
- `episode_discovery.json`: local episode discovery for batch evaluation.

## Interpretation

Scores on one episode validate the data path and ablation mechanics. They should
not be read as cross-episode generalization. A stronger benchmark should split
by held-out episodes and include more tasks, homes, camera motions, and object
arrangements.

The current public Xperience-10M sample distribution exposes one verified sample
episode. The CLI supports `--data-roots` for multi-episode runs, and
`scripts/discover_xperience_samples.py` records which local episodes are
available before running a batch evaluation.

## Failure Modes

- Chronological labels can become imbalanced near the held-out tail.
- RGB features can overfit scene layout.
- Hand joints can be noisy or missing.
- MLP results can vary more with seed and small sample size.

## Next Work

- Add held-out episode evaluation when more data is available.
- Add calibration plots for predicted confidence.
- Compare lightweight temporal models such as 1D CNNs or GRUs.
