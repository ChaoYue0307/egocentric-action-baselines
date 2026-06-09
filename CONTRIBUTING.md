# Contributing

Thanks for improving this project. The goal is to keep the repo easy to learn
from, easy to run, and honest about what the sample data can and cannot prove.

## Local Checks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
ruff check .
pytest -q
python scripts/run_ablation.py --help
```

## Data Policy

Do not commit raw videos, `annotation.hdf5`, `.rrd` recordings, credentials, or
large generated outputs. Keep sample artifacts small enough to review in git.

## Pull Request Checklist

- Explain the learning or engineering value of the change.
- Add or update tests for behavior changes.
- Update README, cards, reports, or the webpage when user-facing behavior changes.
- Keep examples reproducible with the local sample data contract.
