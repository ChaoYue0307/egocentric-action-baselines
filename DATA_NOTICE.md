# Data Notice

This repository does not include raw Xperience-10M files.

Place a local sample episode on your machine and pass it with `--data-root`:

```bash
export DATA_ROOT=/path/to/xperience-10m-sample
python scripts/run_ablation.py --data-root "$DATA_ROOT"
```

The action baseline expects:

- `annotation.hdf5`
- `fisheye_cam0.mp4`

Generated outputs are written under `outputs/`.
