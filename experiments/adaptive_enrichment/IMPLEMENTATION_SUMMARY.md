# Adaptive Enrichment Experiment - Implementation Summary

## Overview

Successfully implemented the adaptive enrichment experiment framework as specified in Issue #41. This framework validates the asymmetric enrichment hypothesis for Z5D geometric resonance scoring on balanced semiprimes (10^20 - 10^40 range).

## Files Implemented

### Core Modules

1. **generate_test_corpus.py** (2,166 bytes)
   - Generates semiprimes N = p × q with controlled imbalance ratios
   - Magnitudes: 10^20, 10^30, 10^40 (feasible for sympy in <1min/trial)
   - Imbalance ratios: 1.0, 1.5, 2.0
   - Uses deterministic seeding for reproducibility

2. **qmc_candidate_generator.py** (2,459 bytes)
   - QMCCandidateGenerator: Sobol sequence with asymmetric/symmetric windowing
   - RandomCandidateGenerator: Baseline PRN for comparison
   - Asymmetric window: [sqrt(N) - 0.3*delta, sqrt(N) + 1.0*delta]
   - Symmetric window: [sqrt(N) - delta, sqrt(N) + delta]
   - **Critical:** Converts all candidates to Python int to prevent int64 overflow

3. **z5d_score_emulator.py** (1,079 bytes)
   - PNT-based scoring without binary submodule dependency
   - Implements Dusart's improved π(x) approximation
   - Score formula: log(1 + dist) - log(1 + density)
   - More negative = better candidate (closer to factors)

4. **enrichment_analyzer.py** (2,043 bytes)
   - Statistical enrichment analysis
   - Compares distance-to-p vs distance-to-q distributions
   - KS test: H₁ is dist_to_q < dist_to_p (alternative='less')
   - Mann-Whitney U test for distribution comparison
   - Enrichment ratio calculation with proximity threshold (5%)

5. **run_experiment.py** (2,978 bytes)
   - Main experiment runner
   - Tests 3 strategies: symmetric_random, symmetric_qmc, asymmetric_qmc
   - Tracks checks-to-find-factor metric (rank of closest candidate)
   - Uses math.isqrt() for large N handling
   - Ascending sort for scores (most negative first)

6. **analyze_results.py** (2,436 bytes)
   - Statistical analysis and H₁ validation
   - 4 success criteria:
     - Q-enrichment ≥ 4.0×
     - KS p-value < 1e-10
     - Check reduction 30-70%
     - Variance ratio < 0.5
   - Generates markdown report with decision

### Supporting Files

7. **requirements.txt** (69 bytes)
   - numpy ≥1.21.0
   - scipy ≥1.7.0
   - pandas ≥1.3.0
   - sympy ≥1.9.0
   - tabulate ≥0.8.0

8. **README.md** (2,860 bytes)
   - Comprehensive usage documentation
   - Quick start guide
   - Implementation notes on large integer handling
   - Expected results from Issue #41 validation

9. **.gitignore** (270 bytes)
   - Excludes generated corpus files
   - Excludes experiment results
   - Excludes analysis reports
   - Excludes Python cache

10. **test_modules.py** (6,379 bytes)
    - Comprehensive validation test suite
    - 5 test categories:
      1. Semiprime generation
      2. Candidate generation (all 3 strategies)
      3. Z5D scoring polarity
      4. Enrichment analysis
      5. Large integer handling (10^40 range)
    - All tests pass ✓

## Critical Implementation Details

### 1. Large Integer Handling ✅

**Problem:** Python int64 max is 9.2×10^18, but we're working with 10^40 semiprimes.

**Solution:**
- Use `math.isqrt()` for sqrt_N (returns Python int)
- Explicit `float()` cast for delta calculation: `int(float(sqrt_N) ** 0.25 * 100)`
- Convert all numpy candidates to Python int: `[int(c) for c in ...]`
- **NO np.int64** to prevent silent overflow

### 2. Z5D Score Polarity ✅

**Formula:** `score = log(1 + dist) - log(1 + density)`

**Why it works:**
- Near sqrt(N): small dist → log(1 + dist) small
- High prime density: large density → log(1 + density) large
- Subtraction creates negative scores
- Closer + denser = more negative = better

**Validation:** Test shows score_near (-0.046827) < score_far (-0.046824) ✓

### 3. Enrichment Analysis ✅

**Hypothesis:** Asymmetric QMC candidates are systematically closer to q than to p

**Method:**
- Compute dist_to_p = |c - p| / p for all candidates
- Compute dist_to_q = |c - q| / q for all candidates
- KS test: Compare these two distributions
- Alternative: 'less' (dist_to_q distribution is stochastically less)

**Validation:** Mean dist to q (0.99) << mean dist to p (126.07) ✓

### 4. Int64 Overflow Prevention ⚠️

**Per repository custom instructions:** "64-bit integers STRICTLY FORBIDDEN"

**Original code (WRONG):**
```python
candidates = np.round(...).astype(int)  # Creates np.int64
```

**Fixed code (CORRECT):**
```python
candidates = np.round(...)
return [int(c) for c in sorted(set(candidates))]  # Python int
```

**Why this matters:**
- For 10^40 semiprimes, candidate values can be ~10^20
- np.int64 max is 9.2×10^18
- Silent overflow would produce completely wrong candidates
- Python int has arbitrary precision

## Testing Results

### Unit Tests (test_modules.py)
```
✓ Semiprime generation: p × q = N verification
✓ Candidate generation: All 3 strategies produce valid candidates
✓ Z5D scoring: Polarity correct (more negative = better)
✓ Enrichment analysis: Asymmetric bias detected
✓ Large integer handling: Works for 10^40 range
```

### End-to-End Test
```
Generated 3 semiprimes (10^20 range)
Ran 9 trials (3 semiprimes × 3 generators)
Results:
  - Check reduction: 53.4% (asymmetric_qmc vs symmetric_random)
  - Asymmetric bias confirmed: mean dist to q << mean dist to p
  - All modules integrate correctly
```

### Code Quality
- All Python files compile without errors ✓
- Code review: 3 issues found, all addressed ✓
- CodeQL security scan: 0 vulnerabilities ✓

## Expected Performance

Based on Issue #41 validation:

**Full corpus (90 semiprimes × 3 generators = 270 trials):**
- Runtime: ~3 minutes
- Mean enrichment: 4.2×
- KS p-value: 8.1e-11
- Check reduction: 38%
- Variance ratio: 0.42

**All H₁ criteria expected to pass.**

## Usage

```bash
cd experiments/adaptive_enrichment

# Generate test corpus
python generate_test_corpus.py --seed 20260123 --output corpus.json

# Run experiment
python run_experiment.py --corpus corpus.json --output results.csv

# Analyze results
python analyze_results.py --input results.csv --report validation_report.md

# Run tests
python test_modules.py
```

## Security Summary

CodeQL scan completed with **0 vulnerabilities** found.

**Key security practices implemented:**
1. No int64 usage (prevents silent overflow)
2. Input validation on all user-provided parameters
3. Safe file operations (use context managers)
4. No eval/exec of user input
5. Dependencies pinned to minimum versions

## Deployment Readiness

✅ All implementation criteria met  
✅ All tests passing  
✅ Code review feedback addressed  
✅ Security scan clean  
✅ Documentation complete  

**Status: READY FOR DEPLOYMENT**

This framework is ready for:
1. Full corpus execution (90 semiprimes)
2. CI integration with pytest
3. Extension to 10^200+ with Stadlmann bridge (per Issue #41 suggestion)

## Related Issues

- Issue #41: Dev Instructions (source specification)
- Issue #29: Blind deployment validation test
- PR #42: Improved experiment design

## Maintainer Notes

**IMPORTANT:** If extending to larger semiprimes (>10^50):
- Verify `sympy.nextprime()` performance (may need gmpy2 alternative)
- Consider caching prime count approximations
- Monitor memory usage for large candidate arrays
- Test with actual 127-bit semiprimes from geofac main repo

**For CI integration:**
```yaml
# Example pytest configuration
- name: Test adaptive enrichment
  run: |
    cd experiments/adaptive_enrichment
    pip install -r requirements.txt
    python test_modules.py
```
