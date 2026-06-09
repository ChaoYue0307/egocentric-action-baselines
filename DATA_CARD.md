# Data Card

## Source

This project uses one local Xperience-10M sample episode of a pour-over coffee
task. Raw videos and `annotation.hdf5` stay outside the repository.

## Inputs Used

- `fisheye_cam0.mp4` for RGB frame features.
- `annotation.hdf5` for action labels and 3D hand joints.

## Scope

The sample is useful for learning the data flow and evaluation mechanics of
egocentric action recognition. It is not a population-level benchmark.

## Limitations

- Single episode.
- Single environment and task.
- Overlapping windows can make random splits overly optimistic.
- Chronological evaluation is more realistic but still not a held-out-episode test.
