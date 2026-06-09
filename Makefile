PYTHON ?= python3

.PHONY: test help visuals sample pages

test:
	$(PYTHON) -m pytest -q

help:
	$(PYTHON) scripts/run_ablation.py --help

visuals:
	$(PYTHON) scripts/render_action_visuals.py

sample:
	$(PYTHON) scripts/run_ablation.py --data-root ../data/sample/xperience-10m-sample --output-dir outputs/sample_ablation --split-strategy chronological --model classical

pages:
	@echo "https://chaoyue0307.github.io/egocentric-action-baselines/"
