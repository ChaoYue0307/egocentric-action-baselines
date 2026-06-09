PYTHON ?= python3

.PHONY: test help visuals pages

test:
	$(PYTHON) -m pytest -q

help:
	$(PYTHON) scripts/run_ablation.py --help

visuals:
	$(PYTHON) scripts/render_action_visuals.py

pages:
	@echo "https://chaoyue0307.github.io/egocentric-action-baselines/"
