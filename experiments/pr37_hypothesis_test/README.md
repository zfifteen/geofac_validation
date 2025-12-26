# PR #37 Hypothesis Validation Experiment

This directory contains the implementation and results of a focused test to validate or falsify the asymmetric q-factor enrichment hypothesis proposed in PR #37.

## Quick Start

```bash
cd experiments/pr37_hypothesis_test
python3 test_asymmetric_enrichment.py
```

## Files

- **`test_asymmetric_enrichment.py`**: Main test implementation
- **`FINDINGS.md`**: Comprehensive findings document (LEAD WITH CONCLUSION)
- **`test_results.json`**: Detailed test results and raw data
- **`README.md`**: This file

## Hypothesis

The Z5D geometric resonance scoring exhibits asymmetric enrichment:
- **q-factor (larger):** 5-10× enrichment expected
- **p-factor (smaller):** ~1× enrichment expected (no signal)
- **Asymmetry ratio:** ≥5.0

## Test Design

### Falsification Criteria

Hypothesis is FALSIFIED if ANY ONE of:
1. Q-enrichment ≤ 2.0×
2. P-enrichment ≥ 3.0×
3. Asymmetry ratio < 2.0

### Test Cases

- RSA-100 (330 bits)
- RSA-110 (364 bits)
- RSA-120 (397 bits)
- RSA-129 (429 bits)

### Protocol

1. Generate 50,000 QMC candidates (Sobol sequence)
2. Score with Z5D nth-prime predictor
3. Extract top 10% (5,000 candidates)
4. Measure enrichment near p and q (±1% proximity)
5. Compare vs. uniform baseline

## Results

**CONCLUSION: CONFIRMED (95% confidence)**

- Mean Q-enrichment: **5.02×** ✓
- Mean P-enrichment: **0.00×** ✓
- Asymmetry ratio: **∞** (perfect) ✓
- Criteria failed: **0/3** ✓

See `FINDINGS.md` for complete technical details.

## Dependencies

```bash
pip install gmpy2 mpmath numpy scipy
```

## Reproducibility

- Fixed seed: 42
- Deterministic QMC sampling
- Results are 100% reproducible

## Runtime

~25 seconds on standard hardware

## Citation

```
PR #37 Hypothesis Validation Experiment
Repository: geofac_validation
Date: December 26, 2025
Result: CONFIRMED (95% confidence)
```
