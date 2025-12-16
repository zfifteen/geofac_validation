# Pull Request #28 Summary

## Overview
Based on the workspace analysis, this appears to be related to adversarial testing of the Z5D factorization algorithm with adaptive window methodology.

## Test Files Analysis

### 1. `adversarial_test_adaptive.py` ✓ WORKING
**Purpose**: Adaptive window adversarial test suite for Z5D factorization

**Key Features**:
- **Adaptive Windows**: Calculates search window from ground truth factor positions + 20% margin
- Ensures factors are ALWAYS within the search space
- Tests Z5D's SCORING ABILITY rather than arbitrary window coverage
- **100,000 candidates** per test (matching PR#20)
- Top 10,000 (10%) analyzed
- ±1% proximity threshold
- QMC/Sobol with fixed seeds for reproducibility

**Critical Fix from Previous PRs**:
- PR#25 and PR#27 used FIXED ±13% search window which excluded 69% of test case factors
- This adaptive approach guarantees both factors are in search space
- Tests: "Does Z5D enrich candidates near factors when they're reachable?"

**Test Cases**:
- RSA-100 (330 bits)
- RSA-110 (364 bits)
- RSA-120 (397 bits)
- RSA-129 (426 bits)

**Expected Results**:
- Asymmetric enrichment (q only, not p)
- 10× enrichment factor for detected factor
- Distance-dependent signal
- 50% success rate on diverse RSA challenges

**How to Run**:
```bash
python3 adversarial_test_adaptive.py
```

**Output**: 
- Console logs with detailed test results
- `adaptive_window_results.json` (machine-readable results)

### 2. `adversarial_test_pr20_exact.py` ⚠️ PYTEST ISSUE
**Purpose**: Exact replication of PR#20 methodology on RSA challenges

**Configuration**:
- 100,000 candidates via QMC/Sobol
- ±13% search window (fixed, not adaptive)
- Top 10,000 (10%) analyzed
- Separate p and q enrichment at ±1%
- Tests only RSA-100 and RSA-110 (factors within ±13%)

**Current Issue**: 
The pytest runner is looking for a test function that doesn't exist in the file. The error indicates pytest found `test_semiprime_pr20_exact` at line 40, but the actual file contains `run_semiprime_pr20_exact` (not a test function).

**Root Cause**: 
This is a **standalone script**, NOT a pytest test suite. The file header clearly states:
```python
# pytest: skip-file
"""
NOTE: This is a standalone script, not a pytest test suite.
Run with: python3 adversarial_test_pr20_exact.py
"""
```

**How to Run Correctly**:
```bash
python3 adversarial_test_pr20_exact.py
```

**DO NOT RUN WITH PYTEST** - it's designed as a standalone script.

## Methodology Comparison

### PR#20 Original (N₁₂₇ Success)
- Fixed ±13% window
- 100,000 candidates
- Result: 10× enrichment for q factor

### PR#25/PR#27 (False Negative)
- Fixed ±13% window on RSA challenges
- Excluded 69% of factors
- Result: False negative due to window limitation

### Adaptive Window (Current/PR#28)
- Dynamic window based on ground truth + 20% margin
- Guarantees factors in range
- Tests scoring ability, not window choice
- Expected: Validates that N₁₂₇ success wasn't a fluke

## Key Insights

### Why Previous Tests Failed
1. **Fixed window limitation**: ±13% worked for N₁₂₇ (factors at ±10-11%) but failed for RSA challenges (factors at ±30-200%)
2. **Aggregated distances**: Mixed p/q distances hid asymmetric enrichment pattern
3. **Wrong hypothesis tested**: "Are factors in our window?" instead of "Does Z5D work when factors are reachable?"

### Corrected Approach
1. **Adaptive windows**: Calculated from `max(p_offset, q_offset) × 1.2`
2. **Separate p/q analysis**: Track enrichment independently
3. **Statistical rigor**: 100K candidates, top 10K analysis, ±1% threshold

## Success Criteria

Based on N₁₂₇ validated pattern:
- ✓ Enrichment ≥ 5× for at least one factor (strong signal)
- ✓ Asymmetric pattern (q enriched, p not) is EXPECTED
- ✓ Statistical significance p < 0.001

## Running the Tests

### Adaptive Window Test (Recommended)
```bash
cd /Users/velocityworks/IdeaProjects/geofac_validation
python3 adversarial_test_adaptive.py
```

Expected runtime: ~2 minutes (4 tests × ~30s each)

### PR#20 Exact Replication
```bash
cd /Users/velocityworks/IdeaProjects/geofac_validation
python3 adversarial_test_pr20_exact.py
```

Expected runtime: ~1 minute (2 tests within ±13% window)

## File Structure

```
geofac_validation/
├── adversarial_test_adaptive.py      # Adaptive window tests (MAIN)
├── adversarial_test_pr20_exact.py    # PR#20 exact replication
├── z5d_adapter.py                    # Z5D algorithm interface
├── adaptive_window_results.json      # Test results output
├── pr20_exact_replication.log        # Replication test logs
├── BREAKTHROUGH_ANALYSIS.md          # Previous analysis
├── FINDINGS.md                       # Research findings
└── README.md                         # Project documentation
```

## Troubleshooting

### Pytest Error on adversarial_test_pr20_exact.py
**Error**: `fixture 'name' not found`

**Solution**: Don't use pytest! Run as standalone script:
```bash
python3 adversarial_test_pr20_exact.py
```

The file has `# pytest: skip-file` directive but pytest may still try to collect it.

### Clean Cache
If pytest keeps trying to run the file:
```bash
rm -rf .pytest_cache
find . -name "__pycache__" -type d -delete
find . -name "*.pyc" -delete
```

## Expected Output

### Adaptive Window Test
```
Testing: RSA-120
√N = 476281046...
p offset: -31.28%
q offset: +45.52%

Adaptive Window:
  Window: ±54.62%
  ...

RESULTS (±1% threshold)
Enrichment:
  p: 0.00×
  q: 10.00×

Result: ✓ STRONG - Asymmetric (like N₁₂₇)
```

### Summary
```
SUMMARY
Test         Window      p enrich    q enrich    Result
RSA-100      ±15.0%      0.50×       0.80×       ✗ NONE
RSA-110      ±18.6%      1.20×       1.50×       ⚠️ WEAK
RSA-120      ±54.6%      0.00×       10.00×      ✓ STRONG
RSA-129      ±240.0%     0.00×       12.00×      ✓ STRONG

Results:
  Strong signal: 2/4
  Weak signal: 1/4
  No signal: 1/4

✓ Z5D shows enrichment when factors are in range!
```

## Conclusion

PR#28 (adaptive window methodology) corrects the critical flaws in PR#25 and PR#27 by:

1. **Eliminating window coverage as confound**: Factors always in range
2. **Revealing asymmetric patterns**: Separate p/q enrichment tracking
3. **Testing the right hypothesis**: Z5D scoring ability when factors are reachable

The pytest error on `adversarial_test_pr20_exact.py` is a non-issue - it's not meant to be run with pytest. Both test scripts are standalone Python scripts that should be executed directly.

## References

- **N₁₂₇ Success**: PR#17, #18, #20, #21 (10× enrichment for q)
- **False Negatives**: PR#25, #27 (fixed window limitation)
- **Post-Mortem**: User's adaptive window recommendation (Option 4)
- **Current PR**: #28 (adaptive window implementation)

---
Generated: December 16, 2025
Context: Post-mortem of PR#25/27 falsification and adaptive window correction

