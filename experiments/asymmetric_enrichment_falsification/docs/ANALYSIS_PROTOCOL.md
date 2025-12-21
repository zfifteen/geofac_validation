# Analysis Protocol

## Overview

This document specifies the exact analysis procedures for the asymmetric q-factor enrichment falsification experiment.

## Data Collection Protocol

### Phase 1: Baseline Monte Carlo Enrichment

**Objective:** Establish baseline (random) enrichment near factors p and q

**Procedure:**
1. For each semiprime N with known factors (p, q):
   - Calculate √N using arbitrary precision (gmpy2.isqrt)
   - Define search window: [√N - 15%, √N + 15%]
   - Define proximity window: ε = 1% of √N
   
2. Generate candidates (10 trials per semiprime):
   - Method: Uniform random sampling
   - Count: 100,000 candidates per trial
   - Constraint: All candidates must be odd
   - Seed: base_seed + trial_counter (deterministic)
   
3. Measure enrichment:
   - Count candidates within [p - ε, p + ε]
   - Count candidates within [q - ε, q + ε]
   - Calculate baseline density: n_candidates / search_width
   - Calculate expected uniform: baseline_density × 2ε
   - Calculate enrichment ratio: observed / expected
   
4. Record per trial:
   - Semiprime name and metadata
   - Trial number and seed
   - Candidate count near p (count_p)
   - Candidate count near q (count_q)
   - Expected uniform counts
   - Enrichment ratios: E_p, E_q

**Expected baseline:** E_p ≈ 1.0x, E_q ≈ 1.0x (no enrichment)

---

### Phase 2: Z5D Enrichment Measurement

**Objective:** Measure enrichment in top-scoring Z5D candidates

**Procedure:**
1. For each semiprime N with known factors (p, q):
   - Same search window and proximity as Phase 1
   
2. Generate QMC candidates (10 trials per semiprime):
   - Method: Sobol sequence with Owen scrambling
   - Dimensions: 2 (for 106-bit precision mapping)
   - Count: 100,000 candidates per trial
   - Mapping precision: 106-bit fixed-point
   - Constraint: All candidates must be odd
   - Seed: base_seed + trial_counter (deterministic)
   
3. Score candidates with Z5D:
   - For each candidate c:
     - n_est = z5d_n_est(c) using PNT approximation
     - score = compute_z5d_score(c, n_est)
   - Create scored list: [(c₁, score₁), (c₂, score₂), ...]
   
4. Extract top candidates:
   - Sort by score (ascending - lower is better)
   - Extract top 10% (10,000 candidates)
   
5. Measure enrichment in top 10%:
   - Count top candidates within [p - ε, p + ε]
   - Count top candidates within [q - ε, q + ε]
   - Calculate baseline density: 10,000 / search_width
   - Calculate expected uniform: baseline_density × 2ε
   - Calculate enrichment ratios: E_p, E_q
   - Calculate asymmetry ratio: E_q / E_p
   
6. Record per trial:
   - All baseline measurements (same as Phase 1)
   - Z5D enrichment ratios: E_p, E_q
   - Asymmetry ratio
   - QMC method used

**Expected per claim:** E_p ≈ 1.0x, E_q = 5-10x, Asymmetry ≥ 5.0

---

## Statistical Analysis Protocol

### Test 1: Wilcoxon Signed-Rank (Q-Enrichment)

**Null hypothesis:** median(E_q) = 5.0  
**Alternative:** median(E_q) > 5.0  
**Procedure:**
1. Extract all Z5D q-enrichment values from Phase 2
2. Subtract null value: differences = E_q - 5.0
3. Perform Wilcoxon signed-rank test (one-sided)
4. Record: statistic, p-value

**Decision rule:**
- p < 0.01 (Bonferroni-corrected) → Reject H₀, q-enrichment confirmed
- p ≥ 0.01 → Fail to reject H₀, criterion 1 potentially failed

---

### Test 2: Wilcoxon Signed-Rank (P-Enrichment)

**Null hypothesis:** median(E_p) = 1.0  
**Alternative:** median(E_p) ≠ 1.0  
**Procedure:**
1. Extract all Z5D p-enrichment values from Phase 2
2. Subtract null value: differences = E_p - 1.0
3. Perform Wilcoxon signed-rank test (two-sided)
4. Record: statistic, p-value

**Decision rule:**
- p > 0.01 → Fail to reject H₀, p-enrichment appropriately low
- p ≤ 0.01 → Reject H₀, criterion 2 potentially failed

---

### Test 3: Bootstrap Confidence Intervals

**Objective:** Robust CI estimation without distributional assumptions

**Procedure (for E_q and E_p separately):**
1. Extract enrichment values from Phase 2
2. For b = 1 to 10,000:
   - Resample with replacement (n samples)
   - Calculate mean of resample
   - Store bootstrap mean
3. Sort bootstrap means
4. Calculate percentiles:
   - CI_lower = 2.5th percentile
   - CI_upper = 97.5th percentile
5. Record: mean, CI_lower, CI_upper

**Decision rules:**
- Criterion 1: If CI(E_q) upper < 2.0 → FAILED
- Criterion 2: If CI(E_p) lower > 3.0 → FAILED

**Random seed:** 42 for reproducibility

---

### Test 4: Mann-Whitney U Test

**Null hypothesis:** E_q and E_p have same distribution  
**Alternative:** E_q stochastically dominates E_p  
**Procedure:**
1. Extract E_q values (sample 1)
2. Extract E_p values (sample 2)
3. Perform Mann-Whitney U test (one-sided: greater)
4. Calculate effect size (Cohen's d):
   - pooled_std = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]
   - d = (mean(E_q) - mean(E_p)) / pooled_std
5. Record: statistic, p-value, effect_size

**Decision rule:**
- p < 0.01 AND d > 1.5 → Asymmetry confirmed
- Otherwise → Criterion 3 potentially failed

---

### Test 5: Levene's Test (Scale-Invariance)

**Null hypothesis:** Equal variances across bit-length ranges  
**Alternative:** Variances differ (not scale-invariant)  
**Procedure:**
1. Group E_q values by bit-length range:
   - Small: 64-128 bits
   - Medium: 128-192 bits
   - Large: 192-256 bits
   - RSA-like: 256-384 bits
   - Extreme: 384-426 bits
2. Perform Levene's test for variance homogeneity
3. Record: statistic, p-value

**Decision rule:**
- p > 0.01 → Fail to reject H₀, variances homogeneous (scale-invariant)
- p ≤ 0.01 → Reject H₀, criterion 4 potentially failed

---

### Test 6: Replication Across Ranges (Criterion 4)

**Objective:** Verify pattern replicates across bit ranges

**Procedure:**
1. For each bit-length range:
   - Calculate mean(E_q) for that range
   - Check if mean(E_q) ≥ 5.0 (strong signal)
2. Count ranges with strong signal
3. Determine threshold: min(3, total_ranges - 2)
4. Compare count vs threshold

**Decision rule:**
- Count ≥ threshold → Criterion 4 PASSED
- Count < threshold → Criterion 4 FAILED

---

## Falsification Decision Matrix

**Input:** Results from Tests 1-6

**Procedure:**
1. Evaluate each criterion:
   - **Criterion 1:** Q-enrichment ≤ 2.0x OR CI(E_q) upper < 2.0
   - **Criterion 2:** P-enrichment ≥ 3.0x OR CI(E_p) lower > 3.0
   - **Criterion 3:** Asymmetry ratio < 2.0
   - **Criterion 4:** Strong signal in < threshold ranges
   
2. Count failed criteria

3. Make decision:
   ```
   IF criteria_failed ≥ 2:
       decision = FALSIFIED
       confidence = 0.95 if criteria_failed ≥ 3 else 0.85
   ELIF criteria_failed == 1:
       decision = PARTIALLY_CONFIRMED
       confidence = 0.70
   ELIF criteria_failed == 0 AND p(Wilcoxon_q) < 0.01 AND p(Mann_Whitney) < 0.01:
       decision = CONFIRMED
       confidence = 0.95
   ELSE:
       decision = INCONCLUSIVE
       confidence = 0.50
   ```

4. Generate interpretation text

---

## Visualization Protocol

### Plot 1: Enrichment Comparison

**Type:** Box plots  
**Data:** Baseline E_p, Baseline E_q, Z5D E_p, Z5D E_q  
**Reference lines:**
- y = 1.0 (no enrichment)
- y = 2.0 (Q falsification threshold)
- y = 3.0 (P falsification threshold)
- y = 5.0 (claimed minimum)

**Purpose:** Visual comparison of baseline vs Z5D enrichment

---

### Plot 2: Asymmetry Distribution

**Type:** Histogram  
**Data:** All asymmetry ratios (E_q / E_p)  
**Reference lines:**
- x = 2.0 (falsification threshold)
- x = 5.0 (claimed minimum)
- x = mean(asymmetry) (observed)

**Purpose:** Show distribution and consistency of asymmetric pattern

---

### Plot 3: Confidence Intervals

**Type:** Forest plot  
**Data:** Bootstrap CIs for E_p and E_q  
**Elements:**
- Horizontal lines: CI range
- Diamond markers: mean estimate
- Reference lines at thresholds

**Purpose:** Uncertainty quantification

---

### Plot 4: Enrichment by Bit Range

**Type:** Bar charts (2 panels)  
**Data:** Mean E_q and E_p per bit range  
**Error bars:** Standard deviation  
**Reference lines:** Thresholds (2.0, 3.0, 5.0)

**Purpose:** Test scale-invariance (Criterion 4)

---

### Plot 5: Summary Report

**Type:** Text file  
**Content:**
- Decision and confidence
- Primary metrics with CIs
- Statistical test results
- Criterion pass/fail status
- Interpretation text

**Purpose:** Human-readable summary

---

## Quality Controls

### Data Integrity

1. **No missing trials:** All semiprimes must have exactly 10 baseline + 10 Z5D trials
2. **Candidate counts:** All trials must have exactly 100,000 candidates
3. **Odd constraint:** All candidates must be odd (verify random sample)
4. **Range constraint:** All candidates in [search_min, search_max]

### Computational Correctness

1. **Arbitrary precision:** No int64 overflow (verify bit_length < 64)
2. **Primality verification:** Spot-check 5% of generated primes
3. **Multiplication check:** Verify p × q = N for all semiprimes
4. **Square root check:** Verify p < √N < q for all

### Statistical Validity

1. **Seed reproducibility:** Re-run 3 trials, verify identical results
2. **Bootstrap stability:** Compare 10k vs 5k resamples, verify CI overlap
3. **Distribution checks:** QQ plots for normality (informative, not required)

---

## Reporting Requirements

### Minimal Reporting

For publication/sharing, include:
1. Falsification decision with confidence
2. Primary metrics (E_p, E_q, asymmetry) with 95% CIs
3. Statistical test results (p-values, effect sizes)
4. Criterion pass/fail summary
5. Interpretation text

### Full Reporting

For reproducibility, archive:
1. Complete JSON results (baseline + Z5D + decision)
2. All visualizations (PNG + source data)
3. Experiment metadata (timestamps, seeds, versions)
4. Configuration files (YAML)
5. Git commit hash

---

## Deviation Handling

### If experiment fails to complete:

**Partial data:**
- If ≥80% of trials complete: Analyze available data, note incompleteness
- If <80%: Halt, diagnose issue, restart from checkpoint

**Computational errors:**
- Z5D scoring failures: Skip failed candidates, note count in metadata
- If >10% candidates fail scoring: Investigate, potentially abort

**Statistical anomalies:**
- High variance (CV > 50%): Flag as INCONCLUSIVE
- Bimodal distributions: Investigate subgroups, report separately

---

**Version:** 1.0  
**Last Updated:** December 21, 2025  
**Status:** Finalized, experiment ready
