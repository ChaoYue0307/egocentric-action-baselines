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
For serious evaluation, split by time blocks or by held-out episodes.
