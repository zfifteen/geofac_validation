# Adaptive Enrichment Experiment

## Overview

This experiment validates the asymmetric enrichment hypothesis for Z5D geometric resonance scoring on balanced semiprimes (10^20 - 10^40 range). Implements three candidate generation strategies:

1. **Symmetric Random** - Baseline PRN sampling
2. **Symmetric QMC** - Sobol sequence with symmetric window
3. **Asymmetric QMC** - Sobol sequence with q-biased window

## Quick Start

### Generate Test Corpus

```bash
cd experiments/adaptive_enrichment
python generate_test_corpus.py --seed 20260123 --output corpus.json
```

Generates 90 semiprimes (3 magnitudes × 3 imbalance ratios × 10 samples):
- Magnitudes: 10^20, 10^30, 10^40
- Imbalance ratios: 1.0, 1.5, 2.0

### Run Experiment

```bash
python run_experiment.py --corpus corpus.json --output results.csv
```

Tests all 90 semiprimes with 3 generators (270 trials total). Expected runtime: ~3 minutes.

### Analyze Results

```bash
python analyze_results.py --input results.csv --report validation_report.md
```

Generates statistical analysis and H₁ validation report.

## H₁ Success Criteria

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Q-enrichment | ≥4.0× | Asymmetric QMC shows 4× concentration near q |
| KS p-value | <1e-10 | Distance distributions significantly different |
| Check reduction | 30-70% | Asymmetric QMC reduces checks vs baseline |
| Variance ratio | <0.5 | QMC variance < 50% of PRN variance |

## Files

- `generate_test_corpus.py` - Semiprime generator with controlled imbalance
- `qmc_candidate_generator.py` - Sobol and PRN candidate generators
- `z5d_score_emulator.py` - PNT-based scoring proxy
- `enrichment_analyzer.py` - Statistical enrichment metrics
- `run_experiment.py` - Main experiment runner
- `analyze_results.py` - H₁ validation analysis
- `requirements.txt` - Python dependencies

## Implementation Notes

### Large Integer Handling

**CRITICAL:** Uses Python `int` and `math.isqrt()` for arbitrary precision. No `np.int64` to prevent silent overflow on 10^40 semiprimes.

### Z5D Score Polarity

More negative = better candidate. Score formula:
```
score = log(1 + distance_to_sqrt_N) - log(1 + prime_density)
```

Candidates near sqrt(N) with high prime density get most negative scores.

### Asymmetric Window Bias

Default window: `[sqrt(N) - 0.3*delta, sqrt(N) + 1.0*delta]`

Creates 3.3:1 candidate distribution favoring region above sqrt(N) where q resides.

## Expected Results

Based on validation in Issue #41:

- **Mean enrichment:** 4.2×
- **KS p-value:** 8.1e-11
- **Check reduction:** 38%
- **Variance ratio:** 0.42

All H₁ criteria expected to pass, supporting asymmetric enrichment hypothesis.

## Dependencies

Install with:
```bash
pip install -r requirements.txt
```

Requires:
- numpy ≥1.21.0
- scipy ≥1.7.0
- pandas ≥1.3.0
- sympy ≥1.9.0
