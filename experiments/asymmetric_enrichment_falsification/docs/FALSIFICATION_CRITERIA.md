# Falsification Criteria

## Overview

This document details the four falsification criteria used to evaluate the asymmetric q-factor enrichment hypothesis. The hypothesis is **falsified** if **ANY ONE** of these criteria fails (per original specification).

## Primary Hypothesis

**Claim:** Z5D geometric resonance scoring provides:
- **5-10× enrichment** near the larger factor q (positioned above √N)
- **~1× enrichment** near the smaller factor p (positioned below √N)
- **Asymmetry ratio ≥5.0** (q-enrichment / p-enrichment)
- **Scale-invariant** pattern across 128-426 bit range

## Falsification Criteria

### Criterion 1: Q-Enrichment Too Low

**Threshold:** Mean q-enrichment ≤ 2.0× over 10 trials

**Rationale:**
- Claim states 5-10× enrichment near q
- Conservative falsification: accept ≥2× as weak signal
- ≤2× indicates no meaningful enrichment

**Measurement:**
- Calculate mean(E_q) across all trials
- Check 95% bootstrap CI: upper bound must exceed 2.0×
- If mean ≤ 2.0 OR CI upper bound < 2.0 → criterion FAILED

**Statistical test:**
- Wilcoxon signed-rank: H₀: median(E_q) = 5.0 vs H₁: median(E_q) > 5.0
- Required: p < 0.01 (Bonferroni corrected)

**Example:**
```
Claimed: E_q = 7.5x ± 1.2x
Measured: E_q = 1.8x ± 0.4x  → CRITERION FAILED (too low)
Measured: E_q = 3.5x ± 0.8x  → CRITERION PASSED (weak but detectable)
```

---

### Criterion 2: P-Enrichment Too High

**Threshold:** Mean p-enrichment ≥ 3.0× over 10 trials

**Rationale:**
- Claim states ~1× enrichment near p (no signal)
- Conservative falsification: accept up to 3× as noise
- ≥3× indicates symmetric enrichment, contradicting asymmetry claim

**Measurement:**
- Calculate mean(E_p) across all trials
- Check 95% bootstrap CI: lower bound must be below 3.0×
- If mean ≥ 3.0 OR CI lower bound > 3.0 → criterion FAILED

**Statistical test:**
- Wilcoxon signed-rank: H₀: median(E_p) = 1.0 vs H₁: median(E_p) ≠ 1.0
- Required: p > 0.01 (no significant deviation from 1.0)

**Example:**
```
Claimed: E_p = 1.1x ± 0.3x
Measured: E_p = 4.2x ± 1.1x  → CRITERION FAILED (too high, asymmetry lost)
Measured: E_p = 1.8x ± 0.5x  → CRITERION PASSED (acceptable baseline)
```

---

### Criterion 3: Asymmetry Ratio Too Low

**Threshold:** Mean asymmetry ratio < 2.0

**Rationale:**
- Claim states E_q / E_p ≥ 5.0 (strong asymmetry)
- Conservative falsification: accept ≥2.0 as weak asymmetry
- <2.0 indicates symmetric or reversed pattern

**Measurement:**
- Calculate asymmetry_ratio = E_q / E_p for each trial
- Calculate mean across trials (excluding inf values)
- If mean < 2.0 → criterion FAILED

**Statistical test:**
- Mann-Whitney U: tests if E_q distribution stochastically dominates E_p
- Required: p < 0.01 AND Cohen's d > 1.5 (large effect)

**Example:**
```
Claimed: Asymmetry = 6.8 (strong q-bias)
Measured: Asymmetry = 1.3  → CRITERION FAILED (symmetric)
Measured: Asymmetry = 3.2  → CRITERION PASSED (detectable asymmetry)
```

---

### Criterion 4: Pattern Fails Across Bit Ranges

**Threshold:** Signal present in < 3 bit-length ranges

**Rationale:**
- Claim states scale-invariant pattern across all ranges
- Falsification if pattern only works in specific regimes
- Requires replication across diverse scales

**Measurement:**
- Group trials by bit-length range (Small, Medium, Large, RSA-like, Extreme)
- For each range: calculate mean(E_q)
- Count ranges where mean(E_q) ≥ 5.0× (strong signal)
- If count < min(3, total_ranges - 2) → criterion FAILED

**Statistical test:**
- Levene's test for variance homogeneity across ranges
- Required: p > 0.01 (no significant variance differences)

**Example:**
```
Ranges tested: 5 (Small, Medium, Large, RSA-like, Extreme)
Strong signal in: 1 range (Medium only)  → CRITERION FAILED (not scale-invariant)
Strong signal in: 4 ranges (all except Extreme)  → CRITERION PASSED (replicates)
```

---

## Decision Rules

### FALSIFIED

**Condition:** ≥1 criterion failed (per original specification: "any of the following conditions")

**Confidence:**
- 2+ criteria failed: 95% confidence
- 1 criterion failed: 85% confidence

**Interpretation:** Asymmetric enrichment claim not supported by data

**Action:** Publish negative result, investigate alternative hypotheses

---

### CONFIRMED

**Condition:** 0 criteria failed AND all statistical tests pass

**Confidence:** 95%

**Requirements:**
- Wilcoxon (q): p < 0.01
- Wilcoxon (p): p > 0.01
- Mann-Whitney: p < 0.01, d > 1.5
- Levene: p > 0.01

**Interpretation:** Asymmetric enrichment hypothesis validated

**Action:** 
- Proceed to mechanism investigation
- Deployment readiness assessment
- Publication of positive result

---

### INCONCLUSIVE

**Condition:** High variance, insufficient statistical power, or ambiguous results

**Confidence:** 50%

**Interpretation:**
- Sample size too small
- High variance obscures signal
- Effect size near threshold boundaries

**Action:**
- Increase sample size (more trials or candidates)
- Tighten experimental controls
- Redesign to reduce variance

---

## Bonferroni Correction

**Multiple comparisons problem:** Testing 4 criteria increases Type I error (false positive)

**Solution:** Bonferroni correction
- Family-wise error rate: α = 0.01
- Per-test significance: α' = 0.01 / 4 = 0.0025

**Application:**
- All p-values compared against 0.0025 for strict significance
- Bootstrap CIs at 99.75% level (instead of 95%)
- Conservative approach reduces false positives

---

## Sensitivity Analysis

### What if exactly 2 criteria fail?

**Decision:** FALSIFIED with 85% confidence

**Rationale:**
- Multiple falsification paths increase robustness
- 2/4 failures unlikely due to chance (binomial p < 0.05)
- Conservative: prefer false negative over false positive

### What if criterion failures are marginal?

**Example:** E_q = 2.1× (just above 2.0× threshold)

**Decision:** Use bootstrap CI to account for uncertainty
- If CI lower bound > 2.0 → criterion PASSED
- If CI upper bound < 2.0 → criterion FAILED
- If CI straddles 2.0 → INCONCLUSIVE, increase sample

### What if only Extreme range fails?

**Decision:** Criterion 4 NOT failed if signal present in 4/5 ranges

**Rationale:**
- 426-bit semiprimes are computationally expensive
- Limited sample at extreme scale is expected
- Pattern replication in 4/5 ranges is sufficient

---

## Validation

These criteria were defined **a priori** (before running experiments) to prevent:
- **Post-hoc rationalization** (moving goalposts after seeing results)
- **P-hacking** (trying multiple tests until one is significant)
- **Confirmation bias** (only reporting favorable outcomes)

**Reproducibility:** All criteria, thresholds, and statistical tests specified in `config/statistical_thresholds.yaml` with version control.

---

**Version:** 1.0  
**Last Updated:** December 21, 2025  
**Status:** Finalized, experiment ready
