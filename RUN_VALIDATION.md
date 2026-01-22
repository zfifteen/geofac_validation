# Running Prospective Validation

This guide explains how to run the full prospective validation on all 20 test semiprimes.

## Quick Start

### 1. Run Full Validation (Production Parameters)

```bash
# Run on all 20 semiprimes with full parameters
# WARNING: This may take several hours to complete
python3 run_prospective_validation.py \
    --dataset data/prospective_semiprimes.json \
    --output results/prospective_validation_results.json
```

### 2. Run Quick Test (Reduced Parameters)

```bash
# Quick mode: reduced parameters for faster testing
python3 run_prospective_validation.py \
    --quick \
    --output results/quick_test_results.json
```

### 3. Test on Subset

```bash
# Test on first 5 semiprimes only
python3 run_prospective_validation.py \
    --quick \
    --limit 5 \
    --output results/subset_test.json
```

## Benchmarking Against Other Algorithms

### Run Benchmarks

```bash
# Benchmark trial division and Pollard's Rho
python3 benchmark_algorithms.py \
    --dataset data/prospective_semiprimes.json \
    --validation-results results/prospective_validation_results.json \
    --output results/benchmark_results.json
```

### Available Algorithms

- `trial_division`: Basic trial division (baseline)
- `pollard_rho`: Pollard's Rho algorithm

Note: ECM and GNFS require external libraries (GMP-ECM, CADO-NFS) not available in this environment.

## Generate Analysis Report

```bash
# Generate comprehensive markdown report
python3 generate_validation_report.py \
    --validation results/prospective_validation_results.json \
    --benchmark results/benchmark_results.json \
    --output PROSPECTIVE_VALIDATION_REPORT.md
```

## Expected Runtime

### Quick Mode (--quick flag)
- **10,000 candidates per iteration** (vs 100,000 in production)
- **5 max iterations** (vs 10 in production)
- **~2-5 seconds per semiprime**
- **Total: ~2 minutes for all 20**

### Production Mode (default)
- **50,000-100,000 candidates per iteration** (varies by semiprime size)
- **8-10 max iterations**
- **~5-30 minutes per semiprime** (depends on convergence)
- **Total: 2-10 hours for all 20**

## Success Criteria (Issue #43)

| Tier | Success Rate | Interpretation |
|------|--------------|----------------|
| **Stretch** | ≥75% (15/20) | Excellent evidence |
| **Target** | ≥50% (10/20) | Strong evidence |
| **Minimum** | ≥15% (3/20) | Weak evidence |
| **Below** | <15% | Insufficient evidence |

## Output Files

### Validation Results (`results/prospective_validation_results.json`)

Contains:
- Metadata (timestamp, parameters, dataset)
- Summary statistics (success rate, timing, convergence)
- Detailed results for each semiprime
- Window history for each iteration

### Benchmark Results (`results/benchmark_results.json`)

Contains:
- Algorithm comparison data
- Individual semiprime results per algorithm
- Timing and success/failure information

### Analysis Report (`PROSPECTIVE_VALIDATION_REPORT.md`)

Contains:
- Executive summary
- Results by bit range and offset type
- Detailed success/failure tables
- Algorithm comparison
- Analysis and conclusions
- Recommendations for future work

## Interpreting Results

### Convergence Reasons

- `factor_found`: Factor discovered via GCD test ✅
- `factor_found_final`: Factor found in exhaustive final window test ✅
- `threshold_reached`: Converged to N^(1/4) without finding factor ⚠️
- `max_iterations`: Hit iteration limit without convergence ⚠️
- `invalid_window`: Window became invalid (rare edge case) ❌
- `timeout`: Exceeded time limit ❌

### Window History Analysis

Each iteration record includes:
- `window_center`: Center of search window
- `window_radius`: Half-width of window
- `window_size`: Total window size
- `candidates_tested`: Number of candidates generated
- `best_score`: Best Z5D score in iteration
- `cluster_center`: Computed cluster center for next iteration

Monitor these to understand:
- Is the window narrowing consistently?
- Is the cluster center shifting significantly each iteration?
- Are Z5D scores improving (getting more negative)?

## Troubleshooting

### Script Hangs or Takes Too Long

- Use `--quick` flag to reduce parameters
- Use `--limit N` to test fewer semiprimes
- Check if Z5D scoring is working (should be ~0.1s per candidate batch)

### No Factors Found

This is expected behavior if:
- Using `--quick` mode (insufficient candidates)
- Testing only a few iterations
- Z5D gradient is not strong for that particular semiprime

The validation is designed to assess overall success rate, not guarantee 100% success.

### Out of Memory

- Reduce `candidates_per_iteration` in the script
- Process semiprimes sequentially (already implemented)
- Check window_history size (may grow large for many iterations)

## Next Steps After Validation

1. **Analyze Results:** Review PROSPECTIVE_VALIDATION_REPORT.md
2. **Identify Patterns:** Which offset types succeeded? Which bit ranges?
3. **Debug Failures:** Examine window_history for failed cases
4. **Optimize Parameters:** Adjust zoom_factor, candidates_per_iteration based on findings
5. **Implement Stage 3:** Add Coppersmith handoff for converged windows
6. **Scale Testing:** Try larger semiprimes (256+ bits) if validation succeeds

## Contact

For questions or issues with the validation protocol, refer to:
- **ISSUE_43.md:** Full specification
- **IMPLEMENTATION_SUMMARY.md:** Technical details
- **gradient_zoom.py:** Algorithm source code
