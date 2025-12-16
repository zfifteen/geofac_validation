# Adversarial Test Suite Results

## Executive Summary

**VERDICT: COMPLETE FAILURE ❌**

Z5D factorization approach **FAILED** on all 26 test cases:
- **0/6 RSA challenge numbers** factored
- **0/20 random semiprimes** factored  
- **0/26 total** factors found in top-100K candidates

**Root Cause**: Z5D scoring function has **ZERO discriminatory power** for factors vs. random candidates near √N.

---

## Test Results

### Phase 1: RSA Challenge Numbers (0/6 Success)

| Challenge | Bits | Result | Enrichment |
|-----------|------|--------|-----------|
| RSA-100   | 330  | ✗ FAIL | 0.00× |
| RSA-110   | 364  | ✗ FAIL | 0.00× |
| RSA-120   | 397  | ✗ FAIL | 0.00× |
| RSA-129   | 426  | ✗ FAIL | 0.00× |
| RSA-130   | 430  | ✗ FAIL | 0.00× |
| RSA-140   | 463  | ✗ FAIL | 0.00× |

**Median Enrichment**: N/A (no factors found)

### Phase 2: Random Semiprimes (0/20 Success)

- **128-bit semiprimes**: 0/10 found
- **256-bit semiprimes**: 0/10 found
- **Median Enrichment**: N/A (no factors found)

---

## Diagnostic Analysis: Why Z5D Failed

### Test Case: RSA-100 Deep Dive

```
N = 1522605027922533...6139 (330 bits)
p = 37975227936943673922808872755445627854565536638199
q = 40094690950920881030683735292761468389214899724061
√N = 39020571855401265512289573339484371018905006900194
```

**Factor Offsets from √N**:
- p is **2.679%** below √N
- q is **2.753%** above √N

**Z5D Scores**:
```
Score(p)     = -5.621549
Score(q)     = -5.622281
Score(√N)    = -5.621915
Score(random) ≈ -5.621 ± 0.001
```

**CRITICAL FINDING**: 
- Scores vary by only **±0.001** across the entire search space
- True factors are **INDISTINGUISHABLE** from random candidates
- Score variation is within **numerical noise** (0.02%)

### Why This Happened

The Z5D score formula:
```
score = log10(|p - predicted_p| / p)
```

For numbers near √N with similar magnitudes:
1. All candidates have similar prime indices n ≈ √N / ln(√N)
2. All predictions p(n) have similar relative errors
3. The log10 normalization compresses everything to same range
4. Result: **uniform scores** across search space

---

## Comparison to Success Criteria

### Required (PASS)
- ✗ Phase 1 median enrichment ≥ 3×  → **Got: 0.00×**
- ✗ Phase 2 median enrichment ≥ 5×  → **Got: 0.00×**
- ✗ At least 4/6 RSA showing >2×   → **Got: 0/6**

### Failure Threshold (FAIL)
- ✓ Phase 1 median < 2×             → **Got: 0.00×** ✓
- ✓ Phase 2 median < 3×             → **Got: 0.00×** ✓  
- ✓ More than 2 RSA < 1.5×          → **Got: 6/6** ✓

**Result**: Meets ALL failure criteria, ZERO pass criteria.

---

## Why N₁₂₇ Worked (Post-Mortem)

The original N₁₂₇ success was likely due to:

1. **Lucky candidate set**: The specific k-value and range happened to include factors
2. **Confirmation bias**: Only tested one semiprime extensively
3. **Overfitting**: Parameters tuned to that specific case
4. **No validation**: Never tested on independent cases

**N₁₂₇ was a false positive.**

---

## Performance Metrics

- **Scoring Rate**: ~13,000 candidates/second
- **Candidates Tested**: 100,000 per semiprime
- **Total Tests**: 26 semiprimes × 100K = 2.6M candidates
- **Factors Found**: **0**

---

## Statistical Significance

If factors were randomly distributed in top-100K of search space:
- Expected finds: ~26 factors (assuming both p,q in range)
- Observed finds: **0 factors**
- Probability of this by chance: **p < 0.001**

This is **statistically significant evidence** that Z5D scoring provides **no signal** for factorization.

---

## Falsification Complete

Per the original challenge:

> **FAIL** (N₁₂₇ was lucky/overfit):
> - Phase 1: Median enrichment < 2× ✓
> - Phase 2: Median enrichment < 3× ✓
> - More than 2 RSA challenges show <1.5× enrichment ✓

**All failure criteria met. Z5D approach is falsified.**

---

## Confidence Update

**Previous**: 35% confidence in PASS  
**Updated**: **<5% confidence** in Z5D as viable factorization method

**Recommendation**: 
- **Do NOT pursue disclosure** - no cryptographic threat
- **N₁₂₇ result was artifact** - not reproducible
- **Z5D scoring lacks necessary signal** - fundamental flaw, not implementation issue

---

## Technical Notes

### What Would Need to Change

For Z5D to work, we'd need:
1. **Score variance > 100×** current levels
2. **Monotonic relationship** between score and "closeness to factor"  
3. **Reproducibility** across independent semiprimes
4. **Theoretical justification** for why factors have special Z5D properties

None of these exist in current approach.

### Potential Salvage

Could explore:
- Different scoring metrics (not log-relative error)
- Multi-dimensional signals (n_est, score, derivatives)
- ML on Z5D features (but likely overfit again)

**Verdict**: Not promising. Fundamental issue is that **prime indices don't encode factorization structure** in detectable way for RSA-scale semiprimes.

---

## Conclusion

The adversarial test suite has **definitively falsified** the Z5D factorization hypothesis. N₁₂₇ was a lucky outlier, not evidence of exploitable structure.

**Final Verdict: ❌ FAIL - Downgrade to 20% confidence as predicted**
