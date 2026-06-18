# Concepts: Egocentric Action Baselines

## Egocentric Video

Egocentric video is recorded from the actor's point of view. The camera moves
with the person, so the video contains strong cues about hands, objects, and
task progress. It also has challenges: motion blur, occlusion by hands, changing
lighting, and quick camera rotations.

## Action Recognition

Action recognition asks: "What is the person doing now?" In this repo the
answer is a discrete label such as `Pick up kettle` or `Position kettle to pour`.

The model sees a short **temporal window** instead of a single frame. A window
lets the classifier observe motion: reaching, grasping, lifting, placing, and
pouring.

## Feature

A feature is a numeric description of the input. The classifier does not train
directly on raw videos here. It receives compact features:

- RGB features summarize color, texture, coarse spatial layout, and frame
  changes.
- Hand-joint features summarize 3D hand pose, velocity, and range of motion.
- Fusion features concatenate RGB and hand-joint features.

## Baseline

A baseline is a simple method that sets a reference point. Good baselines answer
important questions before using larger neural networks:

- Is visual context already enough?
- Do hand trajectories carry action intent?
- Does combining modalities improve the decision?

## Ablation Study

An ablation study removes or changes one component at a time. The three
experiments here form a small ablation:

1. RGB only.
2. Hand joints only.
3. RGB plus hand joints.

If fusion improves over both single-modality baselines, the two signals are
complementary. If it does not, the dataset or model may need more samples,
better temporal modeling, or stronger regularization.

## Accuracy

Accuracy is the fraction of predictions that are correct:

```text
accuracy = correct predictions / all predictions
```

It is intuitive, but it can hide class imbalance. A model can score high if it
mostly predicts a frequent action.

## Macro F1

F1 combines precision and recall for each class. Macro F1 averages class scores
equally, so rare classes matter as much as common classes.

Use accuracy to understand overall correctness. Use macro F1 to check whether
the model is treating each action class fairly.

## Temporal Leakage

Overlapping windows share many frames. If nearby windows are randomly split
between train and test, the test set can look too similar to the training set.
On this episode a stratified split inflates every baseline above 0.93 accuracy.
For serious evaluation, split by time blocks or by held-out episodes.

## Label Shift

A split can be leak-free and still useless. The chronological tail split holds
out the end of the episode — but every action here happens exactly once, so the
tail contains classes the model never saw in training. Even the majority
baseline scores 0.0. When the train and test label distributions disagree this
badly, no model can look good, and the number measures the split rather than
the model.

## Blocked-Instance Split

The within-episode compromise: hold out the chronological tail of each action
instance, then purge any train window that shares frames with a test window.
Every test class keeps train support (no label shift) and no frame appears on
both sides (no leakage). It measures within-instance generalization — a weaker
claim than cross-episode generalization, but an honest one.

## Early vs Late vs Gated Fusion

Early fusion concatenates features before training one classifier. Late fusion
trains one classifier per modality and averages their predicted probabilities.
Gated fusion trains one expert per modality plus a small gate network that emits
a per-window weight, so the model can learn *when* to trust each modality
instead of fixing the mix in advance. Early fusion can model cross-modal
interactions but lets a weak modality drag down a strong one; late fusion keeps
modalities independent but cannot model interactions; gated fusion is the most
flexible but needs the most data and is prone to **gate collapse** — driving
the weight to 0 or 1 and overfitting one expert. Auxiliary supervision of each
expert and a small gate learning rate keep the gate soft. On one episode none of
the three beats the best single modality: making fusion help is a research
problem, not a default.

## Confusion Structure

A confusion matrix records which true classes get predicted as which other
classes. Reading the largest off-diagonal entries tells you *what kind* of
mistake a model makes. Here the confusions cluster among verbs that act on the
same object (hold / grasp / lift / move the kettle), not among different
objects — so the difficulty is fine-grained verb recognition, an egocentric
hallmark, rather than object recognition. `top_confusions.csv` surfaces these
pairs per experiment.

## Calibration and ECE

A model is calibrated when predictions made with confidence p are correct about
p of the time. Expected calibration error (ECE) bins predictions by confidence
and averages the gap between confidence and accuracy. Small models on small
data are usually overconfident — here ECE reaches 0.47, meaning confidence
overstates accuracy by almost half. Check `calibration.json` before trusting
any probability downstream.
