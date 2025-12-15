#!/bin/bash
set -e

# Get the directory where the script is located (absolute path)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define directories relative to the script location
SRC_DIR="$SCRIPT_DIR/src"
TOOLS_DIR="$SCRIPT_DIR/tools"
DATA_DIR="$SCRIPT_DIR/data"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"

mkdir -p "$DATA_DIR" "$ARTIFACTS_DIR"

echo "=== Building Z5D Adapter ==="
cd "$SRC_DIR"
make
# Note: Makefile outputs binary to ../z5d_adapter (which is SCRIPT_DIR)

SEEDS_FILE="$DATA_DIR/seeds.csv"
if [ ! -f "$SEEDS_FILE" ]; then
    echo "=== Generating Seeds ==="
    python3 "$TOOLS_DIR/generate_qmc_seeds.py" --samples 100 --dimensions 4 --output "$SEEDS_FILE"
fi

echo "=== Running Crosscheck ==="
# Run Geofac to get peaks, pipe to Z5D adapter
# Use a temp file for Geofac output as run_geofac_peaks_mod.py writes to file, not stdout
GEOFAC_OUT="$DATA_DIR/geofac_peaks.jsonl"
FINAL_OUT="$ARTIFACTS_DIR/crosscheck_results.jsonl"

python3 "$TOOLS_DIR/run_geofac_peaks_mod.py" \
    --seeds "$SEEDS_FILE" \
    --output "$GEOFAC_OUT" \
    --scale-min 14 \
    --scale-max 18 \
    --top-k 1000 \
    --num-bins 1000

echo "=== Validating with Z5D Adapter ==="
"$SCRIPT_DIR/z5d_adapter" < "$GEOFAC_OUT" > "$FINAL_OUT"

echo "=== Done ==="
echo "Results saved to $FINAL_OUT"
