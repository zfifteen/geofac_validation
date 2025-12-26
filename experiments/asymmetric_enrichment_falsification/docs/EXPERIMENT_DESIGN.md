# Experiment Design Rationale

## Overview

This experiment is designed to provide a **decisive falsification test** for the asymmetric q-factor enrichment hypothesis in Z5D geometric resonance scoring.

## Design Principles

### 1. Falsifiability First

The experiment is structured around **clear falsification criteria** rather than confirmation bias:

- **Null hypothesis:** Z5D provides no asymmetric enrichment (E_q ~ E_p ~ 1.0)
- **Alternative hypothesis:** Z5D provides 5-10× enrichment for q, ~1× for p
- **Falsification paths:** 4 independent criteria, any 2 failures → hypothesis falsified

This approach follows Karl Popper's principle: a scientific claim must be falsifiable to be meaningful.

### 2. Stratified Sampling

Test set spans 64-426 bits with controlled imbalance levels:

**Rationale:** 
- Tests scale-invariance claim across 6 orders of magnitude
- Controls for compositional bias (balanced vs imbalanced factors)
- Ensures statistical power across different difficulty regimes

### 3. Paired Comparison Design

Each semiprime tested with **both** baseline Monte Carlo and Z5D:

**Rationale:**
- Eliminates confounds from semiprime-specific properties
- Direct comparison: enrichment relative to uniform random baseline
- Controls for "easy" vs "hard" semiprimes

### 4. Multiple Independent Trials

10 trials per semiprime with different random seeds:

**Rationale:**
- Measures variance in enrichment (tests reproducibility)
- Enables bootstrap confidence intervals
- Prevents cherry-picking or selection bias

## Methodology Choices

### Why QMC (Sobol) Instead of Pure Monte Carlo?

Replicates exact methodology from `adversarial_test_adaptive.py` (validated on N₁₂₇):

- Low-discrepancy sampling provides better coverage
- Same method claimed to produce 10× enrichment
- Fair test: use their exact methodology

### Why ±15% Search Window?

Matches default from existing validation experiments:

- Large enough to contain factors for most test cases
- Adaptive expansion for RSA challenges with distant factors
- Statistical significance: 100k samples in ±15% window ≈ 667 candidates per %

### Why ±1% Proximity Window (ε)?

Balances sensitivity vs specificity:

- Too small: miss true enrichment signal due to discretization
- Too large: dilute signal by including non-enriched region
- ±1% of √N ≈ reasonable "close to factor" definition

### Why Top 10% for Z5D Analysis?

Matches PR#20 validated methodology:

- 10k candidates from 100k total
- Claim is enrichment in top-scoring candidates
- Fair test: same percentage as successful N₁₂₇ experiment

## Statistical Rigor

### Bonferroni Correction

Multiple comparisons problem: testing 4 criteria → inflated Type I error

**Solution:** Bonferroni correction α = 0.01 / 4 = 0.0025 per test

### Bootstrap Over Parametric Tests

Enrichment ratios may not follow normal distribution:

- Bootstrap makes no distributional assumptions
- 10,000 resamples provides stable CI estimates
- Robust to outliers and skewness

### Effect Size Requirement

Not just statistical significance—requires **practical significance**:

- Cohen's d > 1.5 for Mann-Whitney U (large effect)
- Enrichment > 5× (not just > 1.0)
- Asymmetry ratio > 5.0 (not just > 1.0)

## Potential Confounds & Controls

### Confound 1: Factor Position Arithmetic

**Risk:** Z5D might exploit knowledge of where factors "should be" based on N

**Control:** 
- Generate decoy factors at random offsets (future extension)
- Verify enrichment doesn't occur at √N ± random %

### Confound 2: QMC Implementation Artifacts

**Risk:** Enrichment from sampling method, not Z5D scoring

**Control:**
- Baseline uses same QMC (Sobol) for fair comparison
- (Future: test with Halton sequence alternative)

### Confound 3: Bit-Length Dependence

**Risk:** Effect only works at specific scales

**Control:**
- Stratified sampling across 64-426 bits
- Levene's test for variance homogeneity
- Criterion 4: must replicate across ≥3 ranges

### Confound 4: Imbalance Artifacts

**Risk:** Enrichment correlated with factor imbalance, not Z5D

**Control:**
- Mix of balanced (0-5%) and imbalanced (40%+) semiprimes
- Regression analysis: enrichment vs imbalance % (future extension)

## Power Analysis

### Sample Size Justification

**Per semiprime:**
- 10 trials × 100k candidates = 1M total samples
- Top 10% = 100k analyzed candidates
- ±1% window = ~2k expected uniform candidates

**Statistical power:**
- Detect 5× enrichment with >99% power at α=0.01
- Detect 2× enrichment with >95% power
- Minimum detectable effect: ~1.5× at 80% power

**Total experiment:**
- 26 semiprimes × 10 trials = 260 measurements
- Sufficient for robust confidence intervals
- Multiple bit-ranges for replication testing

## Assumptions & Limitations

### Assumptions

1. **Primality:** Miller-Rabin with 64 rounds gives <10⁻³⁸ false positive rate
2. **Independence:** Trials with different seeds are independent
3. **Ground truth:** Known factors are correct (verified by p × q = N)

### Limitations

1. **Compute cost:** 426-bit semiprimes expensive to score (100k × 26 × 10 = 26M scorings)
2. **No mechanism testing:** Experiment tests "does it work?" not "why does it work?"
3. **Limited factor diversity:** Only two-prime semiprimes, not general composites

### Out of Scope

- **Blind factorization:** Requires unknown factors (separate experiment)
- **Comparative methods:** Not comparing Z5D to GNFS/QS/ECM
- **Parameter tuning:** Fixed parameters from prior work, no optimization

## Acceptance Criteria

This experiment is accepted as **decisive** if:

1. ✓ Clear falsification criteria defined a priori
2. ✓ Multiple independent falsification paths
3. ✓ Sufficient statistical power (>95%)
4. ✓ Controls for known confounds
5. ✓ Reproducible (fixed seeds, pinned dependencies)
6. ✓ Transparent (all code, data, and analysis published)

## Future Extensions

If hypothesis **confirmed**:
- Mechanism investigation (why asymmetric?)
- Parameter sensitivity analysis
- Blind factorization deployment test

If hypothesis **falsified**:
- Post-hoc analysis of any partial signals
- Alternative enrichment hypotheses
- Lessons for future experiments

---

**Status:** Design finalized, ready for execution  
**Version:** 1.0  
**Date:** December 21, 2025
