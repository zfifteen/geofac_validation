# Makefile for Asymmetric Enrichment Falsification Experiment

.PHONY: help setup run clean results validate test

help:
	@echo "Asymmetric Q-Factor Enrichment Falsification Experiment"
	@echo "========================================================"
	@echo ""
	@echo "Available targets:"
	@echo "  setup      - Install dependencies"
	@echo "  run        - Run complete experiment"
	@echo "  clean      - Remove generated data"
	@echo "  results    - Display experiment results"
	@echo "  validate   - Validate configuration files"
	@echo "  test       - Run quick test with reduced dataset"
	@echo ""

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

run:
	@echo "Starting falsification experiment..."
	python3 run_experiment.py

clean:
	@echo "Cleaning generated data..."
	rm -rf data/benchmark_semiprimes.json
	rm -rf data/results/*.json
	@echo "✓ Data cleaned"

results:
	@if [ -f data/results/falsification_decision.json ]; then \
		echo "Experiment Results:"; \
		echo "=================="; \
		python3 -c "import json; d = json.load(open('data/results/falsification_decision.json')); print(f\"Decision: {d['decision']['decision']}\"); print(f\"Confidence: {d['decision']['confidence']*100:.0f}%\"); print(f\"Q-enrichment: {d['decision']['mean_q_enrichment']:.2f}x\"); print(f\"P-enrichment: {d['decision']['mean_p_enrichment']:.2f}x\")"; \
	else \
		echo "No results found. Run 'make run' first."; \
	fi

validate:
	@echo "Validating configuration files..."
	@python3 -c "import yaml; yaml.safe_load(open('config/semiprime_generation.yaml'))" && echo "✓ semiprime_generation.yaml valid"
	@python3 -c "import yaml; yaml.safe_load(open('config/sampling_parameters.yaml'))" && echo "✓ sampling_parameters.yaml valid"
	@python3 -c "import yaml; yaml.safe_load(open('config/statistical_thresholds.yaml'))" && echo "✓ statistical_thresholds.yaml valid"

test:
	@echo "Running quick test (reduced dataset)..."
	@echo "This will temporarily modify config and restore after test"
	@# Future: implement quick test mode
	@echo "Quick test not yet implemented. Use 'make run' for full experiment."
