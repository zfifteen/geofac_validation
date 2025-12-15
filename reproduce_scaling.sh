#!/bin/bash
set -e

# Reproduce Scaling to 10^1233 Script
# This script reproduces the incremental scaling tests for Geofac-Z5D cross-validation.

echo "=== Reproducing Scaling to 10^1233 ==="
echo "Hardware: $(uname -a)"
echo "Python: $(python3 --version)"
echo "gmpy2: $(python3 -c 'import gmpy2; print(gmpy2.version())' 2>/dev/null || echo 'Not installed')"

# Check for C adapter and build if source available
Z5D_ADAPTER_C=""
Z5D_ADAPTER_PY="python3 z5d_adapter.py"
if [ -f "z5d_adapter" ]; then
    echo "C z5d_adapter available (pre-built)"
    Z5D_ADAPTER_C="./z5d_adapter"
elif [ -f "src/z5d_adapter.c" ] && [ -f "src/Makefile" ]; then
    echo "Building C z5d_adapter..."
    if (cd src && make 2>/dev/null); then
        echo "C adapter built successfully"
        Z5D_ADAPTER_C="./z5d_adapter"
    else
        echo "C adapter build failed"
    fi
fi
echo "Python z5d_adapter.py available (arbitrary precision for large scales)"

# Function to run test for a given scale
run_test() {
    local scale_min=$1
    local scale_max=$2
    local samples=$3
    local desc=$4
    local approx=$5

    echo "=== Testing Scale 10^${scale_min}-10^${scale_max} (${desc}) ==="

    # Generate seeds
    python3 tools/generate_qmc_seeds.py --samples ${samples} --dimensions 4 --output data/seeds_scale_${scale_max}.csv

    # Run Geofac
    echo "Running Geofac..."
    local cmd="python3 tools/run_geofac_peaks_mod.py --seeds data/seeds_scale_${scale_max}.csv --output data/geofac_scale_${scale_max}.jsonl --scale-min ${scale_min} --scale-max ${scale_max} --top-k ${samples} --num-bins ${samples} --max-samples ${samples}"
    if [ "$approx" = "true" ] || [ $scale_max -gt 100 ]; then
        cmd="$cmd --approx"
    fi
    time $cmd

    # Check if output has valid results
    if grep -q '"N":' data/geofac_scale_${scale_max}.jsonl; then
        echo "Geofac produced valid results."

        # Choose adapter: C for small scales (faster), Python for large scales (arbitrary precision)
        # C adapter overflows at uint64 (~10^19 prime index), so use Python for scale > 50
        local adapter
        if [ -n "$Z5D_ADAPTER_C" ] && [ $scale_max -le 50 ]; then
            adapter="$Z5D_ADAPTER_C"
            echo "Running Z5D Adapter (C - fast)..."
        else
            adapter="$Z5D_ADAPTER_PY"
            echo "Running Z5D Adapter (Python - arbitrary precision)..."
        fi
        time $adapter < data/geofac_scale_${scale_max}.jsonl > artifacts/crosscheck_scale_${scale_max}.jsonl

        # Show sample result
        echo "Sample Result:"
        tail -1 artifacts/crosscheck_scale_${scale_max}.jsonl | head -1
    else
        echo "Geofac produced no valid results (likely due to prime generation timeout for large scales)."
        echo "For extreme scales, approximation mode is recommended."
    fi

    echo ""
}

# Test incremental scales (use power-of-2 samples to avoid Sobol warning)
run_test 18 20 8 "Small Scale (True Primes)" false
run_test 99 100 4 "Medium Scale (True Primes)" false
run_test 499 500 8 "Large Scale (Approx)" true
run_test 999 1000 8 "Very Large Scale (Approx)" true
run_test 1232 1233 8 "Extreme Scale (Approx)" true

echo "=== Reproduction Complete ==="
echo "Check data/ and artifacts/ for outputs."
echo "For full 10^1233 with true primes, increase timeout or use distributed computing."