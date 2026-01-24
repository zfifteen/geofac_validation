# CI Integration Guide for Adaptive Enrichment Framework

This document provides instructions for integrating the adaptive enrichment experiment framework into your CI/CD pipeline.

## Prerequisites

- Python 3.8+
- pip package manager

## GitHub Actions Integration

### Basic pytest Integration

Add the following to your `.github/workflows/test.yml`:

```yaml
name: Test Adaptive Enrichment Framework

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest
        pip install -r experiments/adaptive_enrichment/requirements.txt
    
    - name: Run pytest test suite
      run: |
        pytest tests/ -v --tb=short
    
    - name: Run module validation tests
      run: |
        cd experiments/adaptive_enrichment
        python test_modules.py
```

### Full Experiment Validation

To run the full corpus experiment in CI (warning: takes ~3-5 minutes):

```yaml
    - name: Run full experiment validation
      run: |
        cd experiments/adaptive_enrichment
        
        # Generate test corpus (reduced size for CI)
        python generate_test_corpus.py \
          --seed 20260123 \
          --output corpus_ci.json
        
        # Run experiment
        python run_experiment.py \
          --corpus corpus_ci.json \
          --output results_ci.csv
        
        # Analyze results
        python analyze_results.py \
          --input results_ci.csv \
          --report validation_report_ci.md
        
        # Show report
        cat validation_report_ci.md
```

## Local Testing

### Quick Validation

Run the fast unit tests:

```bash
# Install dependencies
pip install pytest
pip install -r experiments/adaptive_enrichment/requirements.txt

# Run pytest suite
pytest tests/ -v --tb=short
```

Expected output:
```
22 passed, 7 warnings in ~1-2s
```

### Module Validation Tests

Run the original validation tests:

```bash
cd experiments/adaptive_enrichment
python test_modules.py
```

Expected output:
```
============================================================
Adaptive Enrichment Experiment - Module Validation Tests
============================================================
✓ ALL TESTS PASSED
============================================================
```

### Full Pipeline Test

Test the complete workflow:

```bash
cd experiments/adaptive_enrichment

# Generate corpus
python generate_test_corpus.py --seed 20260123 --output corpus.json

# Run experiment
python run_experiment.py --corpus corpus.json --output results.csv

# Analyze results
python analyze_results.py --input results.csv --report validation_report.md

# View report
cat validation_report.md
```

## Test Coverage

The pytest suite validates:

### 1. Semiprime Generation (`test_adaptive_enrichment.py::TestSemiprimeGeneration`)
- ✓ Corpus generation with controlled parameters
- ✓ p × q = N correctness
- ✓ Python int output (no int64 overflow)

### 2. Candidate Generation (`test_adaptive_enrichment.py::TestCandidateGeneration`)
- ✓ Symmetric QMC (Sobol sequence)
- ✓ Asymmetric QMC (q-biased window)
- ✓ Random baseline (PRN)
- ✓ Python int output verification

### 3. Z5D Scoring (`test_adaptive_enrichment.py::TestZ5DScoring`)
- ✓ Prime count approximation (PNT)
- ✓ Scoring polarity (more negative = better)
- ✓ Distance-based scoring accuracy

### 4. Enrichment Analysis (`test_adaptive_enrichment.py::TestEnrichmentAnalysis`)
- ✓ Distribution comparison (KS test, Mann-Whitney U)
- ✓ Asymmetric bias detection
- ✓ Proximity threshold calculation

### 5. Large Integer Handling (`test_adaptive_enrichment.py::TestLargeIntegerHandling`)
- ✓ 10^40 range sqrt computation
- ✓ No int64 overflow
- ✓ Arbitrary precision arithmetic

### 6. Workflow Integration (`test_workflow_integration.py`)
- ✓ Corpus generation → JSON output
- ✓ Experiment execution → CSV results
- ✓ Analysis → Markdown report
- ✓ Full pipeline end-to-end

## Expected Metrics

Based on full corpus validation (90 semiprimes × 3 generators):

| Metric | Expected Value | Threshold |
|--------|----------------|-----------|
| Enrichment factor | 4.2× | ≥4.0× |
| KS test p-value | 8.1e-11 | <1e-10 |
| Check reduction | 38% | 30-70% |
| Variance ratio | 0.42 | <0.5 |

**Note:** Small-scale tests (1-10 semiprimes) may not meet these thresholds due to sample size. Use full corpus (90 semiprimes) for validation.

## Performance Benchmarks

### Unit Tests (pytest)
- **Duration:** ~1-2 seconds
- **Coverage:** Core functionality, large integer handling
- **Use case:** Pre-commit checks, CI pull request validation

### Module Validation Tests
- **Duration:** ~1 second
- **Coverage:** Integration between modules
- **Use case:** Development validation

### Full Experiment (90 semiprimes)
- **Duration:** ~3-5 minutes
- **Coverage:** Statistical hypothesis validation
- **Use case:** Release validation, nightly builds

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Ensure dependencies are installed
pip install -r experiments/adaptive_enrichment/requirements.txt

# Ensure tests directory has conftest.py with path configuration
cat tests/conftest.py
```

### Timeout Issues

If corpus generation times out:

```bash
# Reduce timeout or samples per cell
python generate_test_corpus.py \
  --seed 20260123 \
  --output corpus.json
  
# Edit generate_test_corpus.py to adjust:
# - timeout_per_sample (default: 60s)
# - samples_per_cell (default: 10)
```

### Statistical Test Failures

Small corpus sizes may fail H₁ criteria. This is expected:

- **1-10 semiprimes:** Check reduction and enrichment may vary widely
- **30+ semiprimes:** Metrics begin to stabilize
- **90 semiprimes:** Expected to meet all H₁ criteria

## Contact

For issues or questions about the adaptive enrichment framework:
- Open an issue in the repository
- Reference PR #46 for implementation details
- See `experiments/adaptive_enrichment/README.md` for usage documentation
