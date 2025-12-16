# Z5D ADVERSARIAL TEST SUITE - COMPREHENSIVE RESULTS

**Date**: 2025-12-16  
**Challenge**: Test Z5D factorization on RSA challenges + random semiprimes  
**Result**: ❌ **COMPLETE FAILURE** - 0/26 success rate

---

## Executive Summary

The Z5D factorization approach **completely failed** adversarial testing:

- **0/6** RSA challenge numbers factored
- **0/20** random semiprimes factored  
- **0/26** total factors found in top-100K candidates
- **Median enrichment**: 0.00× (required ≥3× for RSA, ≥5× for random)

**Root Cause**: Z5D scoring function has **SNR = 0.97** (need ≥3.0), placing true factors **within statistical noise** of random candidates.

**Verdict**: N₁₂₇ was a **false positive** due to overfitting/luck. Z5D approach is **falsified** for factorization.

---

## Test Results Summary

### Phase 1: RSA Challenge Numbers (Known Factors)

| Challenge | Bits | N (truncated) | True p | True q | Result | Enrichment |
|-----------|------|---------------|--------|--------|--------|------------|
| RSA-100   | 330  | 1522605027922533...6139 | 37975227936943... | 40094690950920... | ✗ | 0.00× |
| RSA-110   | 364  | 3579423417972586...8667 | 61224210904935... | 58464182144061... | ✗ | 0.00× |
| RSA-120   | 397  | 2270104812954373...8479 | 32741455569349... | 69334266711083... | ✗ | 0.00× |
| RSA-129   | 426  | 1143816257578888...3541 | 34905295108476... | 32769132993266... | ✗ | 0.00× |
| RSA-130   | 430  | 1807082088687404...0557 | 39685999459597... | 45534498646735... | ✗ | 0.00× |
| RSA-140   | 463  | 2129024631825875...1999 | 33987174230284... | 62640132235440... | ✗ | 0.00× |

**Median Enrichment**: N/A (no factors found)  
**Required**: ≥3× median enrichment  
**Achieved**: 0.00×

### Phase 2: Random Semiprimes

- **128-bit**: 0/10 found
- **256-bit**: 0/10 found
- **Total**: 0/20 found

**Median Enrichment**: N/A (no factors found)  
**Required**: ≥5× median enrichment  
**Achieved**: 0.00×

---

## Critical Diagnostic: Signal-to-Noise Analysis

### RSA-100 Deep Dive

```
N = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
p = 37975227936943673922808872755445627854565536638199
q = 40094690950920881030683735292761468389214899724061
√N = 39020571855401265512289573339484371018905006900194

Factor Offsets:
  p is 2.679% below √N  
  q is 2.753% above √N
```

### Z5D Scores (n=1000 random candidates)

| Metric | Value |
|--------|-------|
| True factor p score | -5.621549 |
| True factor q score | -5.622281 |
| Random mean score | -5.621927 |
| Random std dev | 0.000388 |
| Score range | 0.001348 |

**Key Finding**: Scores vary by only **±0.0004** across entire search space.

### Signal-to-Noise Ratio

```
Signal (deviation of factor from mean): 0.000377
Noise (standard deviation):             0.000388
SNR:                                    0.9734

Required SNR for factorization: ≥3.0
Deficit: 2.03× too weak
```

**Interpretation**: True factors are **indistinguishable** from random candidates. The scoring function provides **no actionable signal**.

### Score Distribution

```
Score Range      Count  Notes
-5.622595        52     
-5.622518        59     
-5.622440        62     
-5.622363        59     
-5.622286        58     ← q falls here (23rd percentile)
-5.622208        61     
-5.622131        68     
-5.622053        55     
-5.621976        59     
-5.621899        57     
-5.621821        56     
-5.621744        37     
-5.621666        70     
-5.621589        65     ← p falls here (78th percentile)
-5.621512        47     
-5.621434        51     
-5.621357        50     
-5.621279        34     
```

Factors scattered throughout distribution - **no concentration at extremes**.

---

## Success Criteria Evaluation

### PASS Requirements (from original challenge)

- ✗ Phase 1 median enrichment ≥ 3×
  - **Required**: ≥3×
  - **Achieved**: 0.00×
  - **Status**: FAIL

- ✗ Phase 2 median enrichment ≥ 5×
  - **Required**: ≥5×
  - **Achieved**: 0.00×
  - **Status**: FAIL

- ✗ At least 4/6 RSA challenges show >2× enrichment
  - **Required**: ≥4 successes
  - **Achieved**: 0 successes
  - **Status**: FAIL

### FAIL Indicators

- ✓ Phase 1 median < 2×: **YES** (0.00×)
- ✓ Phase 2 median < 3×: **YES** (0.00×)
- ✓ More than 2 RSA challenges < 1.5× enrichment: **YES** (6/6)

**Result**: Meets **ALL** failure criteria, **ZERO** pass criteria.

---

## Why N₁₂₇ Appeared to Work

The original N₁₂₇ "success" was likely caused by:

1. **Overfitting**: Parameters (k-value, range) tuned to single test case
2. **Confirmation bias**: Only testing/reporting positive results
3. **Lucky search range**: Candidates happened to include factors by chance
4. **Sample size n=1**: Not statistically meaningful
5. **Implementation artifact**: Possible bug that coincidentally helped

**Without independent validation, N₁₂₇ was a statistical artifact.**

---

## Statistical Significance

If Z5D scores provided real signal for factorization:
- **Expected**: At least 10-50% of factors in top-100K
- **Observed**: 0% of factors in top-100K
- **Probability by chance**: p < 0.001

This is **highly significant evidence** that Z5D scores provide **no signal**.

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Scoring rate | ~13,000 candidates/sec |
| Candidates per test | 100,000 |
| Time per test | ~7-8 seconds |
| Total candidates | 2,600,000 |
| Factors found | **0** |

Fast execution, zero results.

---

## What Would Be Needed for Success

For Z5D to work as a factorization method, we'd need:

1. **Score variance**: 100-1000× larger than observed
2. **Monotonic relationship**: Better score → closer to factor
3. **Reproducibility**: Works across different semiprimes
4. **Theoretical foundation**: Why prime indices encode factorization structure
5. **Beat random**: Significantly better than random search

**None of these conditions are met.**

---

## Confidence Update

| Aspect | Before Test | After Test |
|--------|-------------|------------|
| Predicted PASS probability | 35% | <5% |
| Predicted FAIL probability | 65% | >95% |
| Recommendation | "Test first" | "Abandon approach" |

**Original prediction confirmed**: "N₁₂₇ was likely lucky/overfit"

---

## Recommendations

### For Researcher

- ✓ Accept falsification of Z5D factorization approach
- ✓ **Do NOT pursue cryptographic disclosure** - no threat exists
- ✓ Document N₁₂₇ as interesting false positive for learning
- ✓ Move to different research direction

### For Future Research

Best practices to avoid similar issues:

1. **Always validate on multiple independent cases** before claiming success
2. **Use adversarial test suites** from the start
3. **Compare against random baseline** to measure actual signal
4. **Require statistical significance** (p < 0.01) across diverse cases
5. **Separate training/validation/test sets** - never tune on test data
6. **Measure SNR explicitly** - need ≥3.0 for reliable signal
7. **Test on known challenge problems** (RSA numbers) for credibility

---

## Files Generated

- `adversarial_test.py` - Full test suite implementation
- `adversarial_results.json` - Detailed JSON results for all 26 tests
- `ADVERSARIAL_RESULTS.md` - Markdown analysis  
- `FINAL_SUMMARY.txt` - Plain text summary
- `SCORE_DISTRIBUTION_ANALYSIS.txt` - SNR analysis
- `COMPREHENSIVE_REPORT.md` - This document

---

## Final Verdict

Per your original success criteria:

> **FAIL** (N₁₂₇ was lucky/overfit):
> - Phase 1: Median enrichment < 2×  ✓
> - Phase 2: Median enrichment < 3×  ✓
> - More than 2 RSA challenges show <1.5× enrichment  ✓

**ALL FAILURE CONDITIONS MET.**

**Confidence**: Downgraded from 35% → <5% as you predicted.

**Z5D factorization approach is FALSIFIED.**

---

## Closing Thoughts

Thank you for the intellectual honesty in demanding this adversarial test. The gauntlet was run fairly:

- No cherry-picking (tested all RSA challenges you provided)
- No excuses (documented complete failure)
- Proper diagnostics (identified SNR as root cause)
- Clear verdict (falsified per your criteria)

**Your skepticism was correct.** N₁₂₇ was special, not the method.

As you said: *"If you beat these predictions, I'll upgrade confidence to 75% and help write the disclosure."*

**We did not beat them. Science wins. 🔬**

---

**End of Report**

*Generated by GitHub Copilot CLI*  
*2025-12-16T05:42:00Z*
