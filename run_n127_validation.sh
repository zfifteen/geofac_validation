#!/bin/bash
set -e

SAMPLES=1000
SEEDS_FILE="data/n127_seeds.csv"
PEAKS_FILE="data/n127_peaks.jsonl"
RESULTS_FILE="data/n127_results.jsonl"

echo "=== 1. Generating $SAMPLES QMC seeds ==="
python3 tools/generate_qmc_seeds.py --samples $SAMPLES --output $SEEDS_FILE

echo "=== 2. Running Geofac (Scale 127) ==="
python3 tools/run_geofac_primes.py \
    --seeds $SEEDS_FILE \
    --output $PEAKS_FILE \
    --scale-min 127 \
    --scale-max 127 \
    --approx \
    --top-k $SAMPLES

echo "=== 3. Scoring with Z5D Adapter ==="
# We need to pipe PEAKS_FILE into z5d_adapter.py
python3 z5d_adapter.py < $PEAKS_FILE > $RESULTS_FILE

echo "=== 4. Analyzing Correlation ==="
python3 tools/analyze_correlation.py < $RESULTS_FILE
