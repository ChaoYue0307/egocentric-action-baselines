PYTHON ?= python3

.PHONY: test help visuals sample split-comparison discover pages

test:
	$(PYTHON) -m pytest -q

help:
	$(PYTHON) scripts/run_ablation.py --help

visuals:
	$(PYTHON) scripts/render_action_visuals.py

sample:
	$(PYTHON) scripts/run_ablation.py --data-root ../data/sample/xperience-10m-sample --output-dir outputs/sample_ablation --model classical --seeds 5 --max-windows 0

split-comparison:
	$(PYTHON) scripts/run_ablation.py --data-root ../data/sample/xperience-10m-sample --output-dir outputs/split_comparison/chronological --split-strategy chronological --model classical --max-windows 0
	$(PYTHON) scripts/run_ablation.py --data-root ../data/sample/xperience-10m-sample --output-dir outputs/split_comparison/stratified --split-strategy stratified --model classical --max-windows 0

discover:
	$(PYTHON) scripts/discover_xperience_samples.py --search-root ../data

pages:
	@echo "https://chaoyue0307.github.io/egocentric-action-baselines/"
