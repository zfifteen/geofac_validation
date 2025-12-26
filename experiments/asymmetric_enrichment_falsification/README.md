# Asymmetric Q-Factor Enrichment Falsification Experiment

**Experiment ID:** GEOFAC-ASYM-001  
**Version:** 1.0  
**Date:** December 21, 2025

## Executive Summary

This experiment implements a rigorous falsification test for the claim that Z5D geometric resonance scoring exhibits **asymmetric enrichment** favoring the larger prime factor (q) over the smaller factor (p) in semiprime factorization, with reported **5-10× signal amplification** near q and negligible signal near p.

## Hypothesis Under Test

**Primary Claim:** Z5D scoring demonstrates reproducible asymmetric bias in factor detection, consistently providing 5-10× enrichment near the larger factor q while showing <1× enrichment near the smaller factor p across 128-426 bit semiprimes.

**Falsification Criteria:** The hypothesis is falsified if ANY of:
1. Q-enrichment ≤2× baseline over 10 trials
2. P-enrichment ≥3× baseline over 10 trials  
3. q/p asymmetry ratio <2.0 (expected ≥5)
4. Pattern fails across ≥3 different bit-length ranges

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure z5d_adapter.py is available in repository root
# (automatically handled if running from repo)
```

### Run Complete Experiment

```bash
cd experiments/asymmetric_enrichment_falsification

# Test modules first (optional but recommended)
make test

# Run full experiment
make run
# OR: python3 run_experiment.py
```

This will:
1. Generate stratified semiprime test set (26 semiprimes across 64-426 bits)
2. Run baseline Monte Carlo enrichment (10 trials × 100k candidates each)
3. Run Z5D enrichment measurement (10 trials × 100k candidates each)
4. Perform statistical analysis (Wilcoxon, Bootstrap CI, Mann-Whitney U)
5. Generate visualizations (enrichment plots, CIs, asymmetry distributions)
6. Make falsification decision
7. Save all results to `data/results/`

**Expected runtime:** 30-60 minutes (depends on hardware)

### Quick Test (Reduced Dataset)

For faster testing, edit `config/semiprime_generation.yaml` to reduce counts:

```yaml
ranges:
  - name: "Small"
    count: 2  # Reduced from 4
    # ...
```

## Directory Structure

```
experiments/asymmetric_enrichment_falsification/
├── config/
│   ├── semiprime_generation.yaml     # Test set specification
│   ├── sampling_parameters.yaml      # QMC/MC parameters
│   └── statistical_thresholds.yaml   # Falsification criteria
├── data/
│   ├── benchmark_semiprimes.json     # Generated test set
│   └── results/
│       ├── phase1_baseline_mc.json          # Baseline enrichment
│       ├── phase2_z5d_enrichment.json       # Z5D enrichment
│       ├── falsification_decision.json      # Final decision
│       └── experiment_metadata.json         # Provenance
├── src/
│   ├── generate_test_set.py          # Semiprime generation
│   ├── baseline_mc_enrichment.py     # Baseline measurement
│   ├── z5d_enrichment_test.py        # Z5D measurement
│   ├── statistical_analysis.py       # Statistical tests
│   └── visualization.py              # Plot generation
├── docs/
│   ├── EXPERIMENT_DESIGN.md          # Design rationale
│   ├── ANALYSIS_PROTOCOL.md          # Analysis methodology
│   └── FALSIFICATION_CRITERIA.md     # Criteria details
├── Makefile                           # Convenience targets
├── test_modules.py                    # Validation tests
├── requirements.txt                   # Dependencies
├── run_experiment.py                  # Main runner
└── README.md                          # This file
```

## Experimental Design

### Test Set Stratification

| Range | Bit Length | Count | Imbalance | Purpose |
|-------|-----------|-------|-----------|---------|
| Small | 64-128 | 8 | 0-5% | Baseline validation |
| Medium | 128-192 | 8 | 5-20% | Core claim range |
| Large | 192-256 | 6 | 20-40% | Asymmetry amplification |
| RSA-like | 256-384 | 2 | 40% | Cryptographic relevance |
| Extreme | 384-426 | 2 | 20-40% | Scale-invariance test |

**Total:** 26 semiprimes × 10 trials = 260 measurements per phase

### Measurement Protocol

#### Phase 1: Baseline Monte Carlo
- Generate 100,000 **uniform random** candidates in [√N - 15%, √N + 15%]
- Measure concentration within ±1% of p and q
- 10 independent trials per semiprime
- Expected: ~1.0× enrichment (no signal)

#### Phase 2: Z5D Geometric Resonance
- Generate 100,000 **QMC candidates** (Sobol sequence)
- Score with Z5D nth-prime predictor
- Extract top 10% by score
- Measure concentration within ±1% of p and q
- Expected per claim: E_q = 5-10×, E_p ~1×

### Statistical Analysis

1. **Wilcoxon Signed-Rank Test**
   - H₀: median(E_q) = 5.0 vs H₁: median(E_q) > 5.0
   - H₀: median(E_p) = 1.0 vs H₁: median(E_p) ≠ 1.0
   - α = 0.01 (Bonferroni correction)

2. **Bootstrap Confidence Intervals**
   - 10,000 resamples
   - 95% CI for enrichment ratios
   - Claim falsified if CI(E_q) includes <2× or CI(E_p) includes >3×

3. **Mann-Whitney U Test**
   - Tests if E_q distribution stochastically dominates E_p
   - Minimum effect size: Cohen's d > 1.5

4. **Levene's Test**
   - Tests variance homogeneity across bit-length ranges
   - Validates scale-invariance claim

## Key Implementation Details

### Arbitrary Precision Arithmetic

**CRITICAL:** All arithmetic uses `gmpy2.mpz` and `mpmath` exclusively. **NO int64/uint64** types to prevent silent overflow on 426-bit semiprimes.

```python
# ✓ CORRECT
N = gmpy2.mpz("large_number_string")
sqrt_N = gmpy2.isqrt(N)

# ✗ WRONG - will overflow silently
N = np.int64(large_number)  # DON'T DO THIS
```

### QMC Sampling Methodology

Replicates exact methodology from `adversarial_test_adaptive.py`:
- Sobol sequence with Owen scrambling
- 2D space with 106-bit fixed-point precision
- Deterministic mapping to search range
- Fixed seeds for reproducibility

### Ground Truth Verification

All semiprimes verified with:
- Miller-Rabin primality test (64 rounds)
- Exact factorization check: p × q = N
- Both factors confirmed prime

## Interpreting Results

### Falsification Decision

Results in `data/results/falsification_decision.json`:

```json
{
  "decision": "CONFIRMED | FALSIFIED | PARTIALLY_CONFIRMED | INCONCLUSIVE",
  "confidence": 0.95,
  "mean_q_enrichment": 7.5,
  "mean_p_enrichment": 1.1,
  "mean_asymmetry_ratio": 6.8,
  "interpretation": "..."
}
```

### Decision Matrix

| Decision | Criteria Failed | Interpretation |
|----------|----------------|----------------|
| **FALSIFIED** | ≥2 | Asymmetric enrichment claim unsupported |
| **PARTIALLY_CONFIRMED** | 1 | Effect exists but weaker than claimed |
| **CONFIRMED** | 0 | Claim validated, all criteria met |
| **INCONCLUSIVE** | - | High variance, insufficient power |

### Confidence Levels

- **CONFIRMED:** 95% confidence, all tests pass
- **FALSIFIED:** 95% confidence if ≥3 criteria fail, 85% if 2 fail
- **PARTIALLY_CONFIRMED:** 70% confidence
- **INCONCLUSIVE:** 50% confidence

## Reproducibility

All results are deterministic and reproducible:

1. **Fixed random seeds:** Base seed 42, incremented per trial
2. **Deterministic QMC:** Sobol initialization with fixed scrambling seed
3. **Exact arithmetic:** Arbitrary precision prevents rounding errors
4. **Version pinning:** All dependencies in `requirements.txt`
5. **Provenance logging:** Git commit, timestamp, parameters saved

To reproduce:
```bash
# Clone repository
git clone https://github.com/zfifteen/geofac_validation
cd geofac_validation/experiments/asymmetric_enrichment_falsification

# Install exact versions
pip install -r requirements.txt

# Run experiment (deterministic output)
python3 run_experiment.py
```

## Files Generated

All results saved to `data/results/`:

- **benchmark_semiprimes.json**: Test set with ground truth factors
- **phase1_baseline_mc.json**: Baseline enrichment measurements (260 trials)
- **phase2_z5d_enrichment.json**: Z5D enrichment measurements (260 trials)
- **falsification_decision.json**: Statistical analysis and decision
- **experiment_metadata.json**: Provenance (timestamp, duration, decision)
- **visualizations/**: PNG plots and summary report
  - enrichment_comparison.png
  - asymmetry_distribution.png
  - confidence_intervals.png
  - enrichment_by_bit_range.png
  - summary_report.txt

## Makefile Targets

```bash
make help       # Show available targets
make setup      # Install dependencies
make test       # Run module validation tests
make validate   # Validate YAML configurations
make run        # Run complete experiment
make results    # Display quick results summary
make clean      # Remove generated data
```

## Citation

If using this experiment framework:

```
Asymmetric Q-Factor Enrichment Falsification Experiment
Experiment ID: GEOFAC-ASYM-001
Repository: https://github.com/zfifteen/geofac_validation
Date: December 21, 2025
```

## Contact & Issues

- **Repository:** https://github.com/zfifteen/geofac_validation
- **Issues:** https://github.com/zfifteen/geofac_validation/issues
- **Related:** Issue #[number], geofac_validation discussions

## License

Same as parent repository (geofac_validation).

---

**Experiment Status:** Ready for execution  
**Last Updated:** December 21, 2025  
**Maintainer:** [Repository owner]
