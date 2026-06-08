# Egocentric Action Baselines

Lightweight first-person action understanding baselines on one complete
Xperience-10M sample episode.

This repo is designed as a learning project for:

- egocentric action and intention understanding,
- hand-object interaction signals,
- task-structure modeling from temporal windows,
- evaluation metrics and per-class analysis,
- ablation studies across RGB, hand joints, and fused features.

Raw Xperience-10M data is not committed. Point the scripts at a local sample
directory:

```bash
/Users/chaoyue/Library/CloudStorage/Dropbox/Ropedia/data/sample/xperience-10m-sample
```

## What It Runs

The main ablation compares three simple baselines:

| Experiment | Input | Model |
| --- | --- | --- |
| `rgb_only` | sampled frames from `fisheye_cam0.mp4` | NumPy softmax classifier |
| `hand_joints_only` | left/right 3D hand joints from `annotation.hdf5` | NumPy softmax classifier |
| `rgb_hand_fusion` | concatenated RGB and hand features | NumPy softmax classifier |

Each run writes:

- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `predictions.csv`
- `model.npz`

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_ablation.py \
  --data-root /Users/chaoyue/Library/CloudStorage/Dropbox/Ropedia/data/sample/xperience-10m-sample \
  --output-dir outputs/sample_ablation \
  --max-windows 240
```

Use `--max-windows 0` to run all labeled windows.

## Notes

This is a single-episode learning repo. High scores should be read as evidence
that the data contract, feature extraction, and evaluation loop are working, not
as cross-episode generalization. The correct research extension is to add many
episodes and split by held-out episode.
