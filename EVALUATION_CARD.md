# Evaluation Card

## Task

Predict the action label for a short egocentric temporal window.

## Metrics

- Accuracy: overall fraction of correct window predictions.
- Macro F1: class-balanced summary that gives each action class equal weight.
- Weighted F1: F1 weighted by class support.

## Current Protocol

The default split is chronological: earlier windows train the classifier and the
last portion of the timeline is used for evaluation. This reduces leakage from
overlapping windows.

## Reading The Result

The numbers show how the baselines behave on one episode. A stronger benchmark
should split by held-out episodes and include more kitchens, camera motions, and
task styles.
