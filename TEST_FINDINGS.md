# Comprehensive Test Suite Findings Report

**Date:** 2026-01-24  
**Tested By:** @copilot  
**Framework:** Adaptive Enrichment Experiment (PR #46)  
**Test Coverage:** 22 pytest tests + 5 module validation tests + end-to-end pipeline

---

## 🎯 CONCLUSIONS

### ✅ What Works
1. **pytest Test Suite (22 tests)** - All pass in ~1.8 seconds
2. **Module Validation Tests (5 tests)** - All pass 
3. **QMC Candidate Generation** - Works correctly for 10^20, 10^30, and 10^40 magnitudes
4. **Z5D Scoring** - Correct polarity and behavior
5. **Enrichment Analysis** - Statistical tests function properly
6. **Corpus Generation** - Successfully generates semiprimes across all magnitudes
7. **Small-scale Pipeline** - Works for magnitudes 10^20 and 10^30

### ❌ Critical Bug Found
**RandomCandidateGenerator fails on 10^40 magnitude semiprimes with int64 overflow**

**Severity:** CRITICAL  
**Impact:** Complete failure of full corpus execution (77 semiprimes)  
**Root Cause:** `numpy.random.default_rng().integers()` uses int64 internally  
**Violation:** Breaks repository's strict "no int64" policy for large integers

---

## 📊 SUPPORTING EVIDENCE

### 1. Test Suite Results

#### pytest Suite (22 tests)
```
Platform: Linux, Python 3.12.3, pytest 9.0.2
Duration: 1.77 seconds
Result: ✅ 22 PASSED, 7 WARNINGS

Test Breakdown:
├── TestSemiprimeGeneration (3 tests) ✅
│   ├── test_generate_small_corpus ✅
│   ├── test_semiprime_validity ✅
│   └── test_python_int_output ✅
├── TestCandidateGeneration (3 tests) ✅
│   ├── test_symmetric_qmc_generation ✅
│   ├── test_asymmetric_qmc_generation ✅
│   └── test_random_generation ✅ (only tests 10^20 range)
├── TestZ5DScoring (3 tests) ✅
│   ├── test_prime_count_approximation ✅
│   ├── test_scoring_polarity ✅
│   └── test_score_is_negative ✅
├── TestEnrichmentAnalysis (2 tests) ✅
│   ├── test_enrichment_computation ✅
│   └── test_asymmetric_bias_detection ✅
├── TestLargeIntegerHandling (3 tests) ✅
│   ├── test_large_integer_sqrt ✅
│   ├── test_candidate_generation_large_n ✅ (only tests QMC)
│   └── test_no_int64_overflow ✅ (only tests QMC)
└── TestExperimentIntegration (1 test) ✅
    └── test_full_pipeline_small_scale ✅
```

#### Workflow Integration Tests (7 tests)
```
Duration: ~0.5 seconds
Result: ✅ 7 PASSED

Test Breakdown:
├── TestWorkflowIntegration (4 tests) ✅
│   ├── test_corpus_generation_workflow ✅
│   ├── test_experiment_workflow ✅
│   ├── test_analysis_workflow ✅
│   └── test_full_pipeline_end_to_end ✅
└── TestExpectedMetrics (3 tests) ✅
    ├── test_enrichment_ratio_range ✅
    ├── test_ks_pvalue_computation ✅
    └── test_check_reduction_calculation ✅
```

#### Module Validation Tests (5 tests)
```
Duration: ~1 second
Result: ✅ ALL TESTS PASSED

1. ✅ Semiprime generation (p × q = N verified)
2. ✅ Candidate generation (all 3 strategies, 100 candidates each)
3. ✅ Z5D scoring (polarity correct: more negative = better)
4. ✅ Enrichment analysis (asymmetric bias detected: dist_to_q << dist_to_p)
5. ✅ Large integer handling (10^40 range, Python int confirmed)
```

### 2. Critical Bug Discovery

#### Bug: RandomCandidateGenerator Int64 Overflow

**Location:** `experiments/adaptive_enrichment/qmc_candidate_generator.py:51`

**Code:**
```python
candidates = self.rng.integers(search_min, search_max + 1, size=n_candidates)
```

**Issue:** `numpy.random.default_rng().integers()` internally uses int64 and cannot handle ranges exceeding `2^63 - 1` (9,223,372,036,854,775,807).

**Failure Case:**
```
Magnitude: 40
N = 2081628363372677110027502176255294714793
sqrt_N = 45624865625804062306
search_max = 45624865625812280954  # EXCEEDS INT64_MAX
int64_max = 9223372036854775807
Overflow: TRUE

Error: ValueError: high is out of bounds for int64
```

**Reproduction:**
```python
from qmc_candidate_generator import RandomCandidateGenerator
from math import isqrt

N_large = 10**40 + 7
sqrt_N_large = isqrt(N_large)

gen = RandomCandidateGenerator(seed=42, asymmetric=False)
candidates = gen.generate_candidates(sqrt_N_large, n_candidates=10)
# >>> ValueError: high is out of bounds for int64
```

**Impact on Full Corpus Run:**
- Corpus generated: 77 semiprimes (expected ~90)
  - 10^20: 26 semiprimes ✅
  - 10^30: 30 semiprimes ✅
  - 10^40: 21 semiprimes ❌ (causes failure)
- Experiment crashes at semiprime #52 (first 10^40 case)
- 3 generators tested per semiprime:
  - symmetric_random: ❌ FAILS on 10^40
  - symmetric_qmc: ✅ WORKS (all magnitudes)
  - asymmetric_qmc: ✅ WORKS (all magnitudes)

**Why Tests Didn't Catch This:**
1. `test_random_generation()` only tests with 10^20 magnitude
2. `test_candidate_generation_large_n()` only tests QMCCandidateGenerator
3. `test_no_int64_overflow()` only tests QMCCandidateGenerator
4. No test specifically validates RandomCandidateGenerator at 10^40 scale

### 3. Full Corpus Execution Results

#### Corpus Generation
```
Command: python generate_test_corpus.py --seed 20260123 --output /tmp/test_corpus.json
Result: ✅ Generated 77 semiprimes (target: 90)

Breakdown:
- Magnitudes: {20, 30, 40}
- Imbalance ratios: 39 unique values (target: {1.0, 1.5, 2.0})
- Note: Some cells have fewer than 10 samples due to timeout/primality failures
```

#### Experiment Execution
```
Command: python run_experiment.py --corpus /tmp/test_corpus.json --output /tmp/test_results.csv
Result: ❌ FAILED at semiprime #52 (magnitude 40)

Error Trace:
File "run_experiment.py", line 67, in main
  trial = run_trial(case, gen, z5d_score)
File "run_experiment.py", line 20, in run_trial
  candidates = generator.generate_candidates(sqrt_N)
File "qmc_candidate_generator.py", line 51, in generate_candidates
  candidates = self.rng.integers(search_min, search_max + 1, size=n_candidates)
ValueError: high is out of bounds for int64
```

### 4. Code Quality Metrics

```
Total Lines of Code: 1,070 lines
├── Test files: 558 lines
│   ├── test_adaptive_enrichment.py: 283 lines
│   ├── test_workflow_integration.py: 258 lines
│   └── conftest.py + __init__.py: 17 lines
└── Source files: 512 lines (experiments/adaptive_enrichment/)

Test Coverage:
- Unit tests: 15/22 (68%) test core functionality
- Integration tests: 7/22 (32%) test workflows
- Code paths covered: ~85% (QMC fully tested, Random partially tested)

Code Review Issues: 3 found, 3 addressed ✅
Security Vulnerabilities: 0 (CodeQL scan) ✅
```

### 5. Warning Analysis

**Sobol Sequence Warning (7 occurrences):**
```
UserWarning: The balance properties of Sobol' points require n to be a power of 2.
  points = self.sobol.random(n_candidates)
```

**Assessment:** Non-critical. Using `n_candidates=512` resolves this, but tests use 10-100 for speed. Does not affect correctness, only statistical properties of QMC sequence.

**Recommendation:** Document in README or use power-of-2 defaults in production.

---

## 🔍 DETAILED TEST ANALYSIS

### Test Category: Large Integer Handling

**Purpose:** Validate arbitrary-precision arithmetic for 10^40 semiprimes

**Tests:**
1. `test_large_integer_sqrt()` ✅
   - Validates `math.isqrt()` for 10^40 + 7
   - Confirms: sqrt(10^40 + 7) = 10^20 (correct)

2. `test_candidate_generation_large_n()` ✅ (INCOMPLETE)
   - Only tests QMCCandidateGenerator
   - **Missing:** RandomCandidateGenerator test
   - **Result:** Bug not caught

3. `test_no_int64_overflow()` ✅ (INCOMPLETE)
   - Only tests QMCCandidateGenerator
   - Checks candidates can exceed int64_max (they can with Python int)
   - **Missing:** RandomCandidateGenerator test
   - **Result:** Bug not caught

### Test Category: Workflow Integration

**Purpose:** Validate end-to-end pipeline

**Tests:**
1. `test_corpus_generation_workflow()` ✅
   - Generates corpus, saves to JSON, validates format
   - Uses magnitude=20 only (safe range)

2. `test_experiment_workflow()` ✅
   - Runs full experiment on magnitude=20 corpus
   - Validates CSV output format and column structure
   - **Missing:** Does not test magnitude=40

3. `test_full_pipeline_end_to_end()` ✅
   - Tests magnitudes=[20, 30] with 2 ratios
   - **Missing:** Does not include magnitude=40
   - **Result:** Pipeline appears to work, but fails on full corpus

---

## 📋 RECOMMENDATIONS

### Immediate (Critical)
1. **Fix RandomCandidateGenerator int64 overflow**
   - Replace `numpy.random.default_rng().integers()` with Python's `random.randint()`
   - Alternative: Implement custom sampling for large ranges
   
2. **Add missing test coverage**
   - Test RandomCandidateGenerator with 10^40 magnitude
   - Add test validating all 3 generators at all 3 magnitudes

### Short-term (Important)
3. **Improve test naming**
   - Rename `test_candidate_generation_large_n()` to clarify it only tests QMC
   
4. **Document Sobol warning**
   - Add note to README about power-of-2 candidate counts
   - Consider defaulting to n_candidates=512 (power of 2)

5. **Expand full corpus tests**
   - Add integration test that includes magnitude=40
   - Consider reducing timeout to avoid long CI runs

### Long-term (Enhancement)
6. **Performance benchmarking**
   - Add timing measurements to tests
   - Validate <200ms execution per semiprime claim

7. **Statistical validation**
   - Run actual 90-semiprime corpus to validate expected metrics
   - Current tests use reduced corpus (< 10 semiprimes)

---

## 🏁 FINAL ASSESSMENT

**Overall Status:** ⚠️ PARTIALLY FUNCTIONAL

**What's Ready:**
- ✅ pytest infrastructure (well-organized, fast)
- ✅ QMC generators (fully functional)
- ✅ Z5D scoring (correct implementation)
- ✅ Enrichment analysis (statistical tests work)
- ✅ Documentation (comprehensive)
- ✅ Security (0 vulnerabilities)

**What's Broken:**
- ❌ RandomCandidateGenerator (int64 overflow on 10^40)
- ❌ Full corpus execution (fails at semiprime #52)
- ❌ Baseline comparison (can't run symmetric_random on 10^40)

**Impact:**
- **H₁ validation:** BLOCKED (cannot test symmetric_random baseline)
- **Expected metrics:** CANNOT VERIFY (need all 3 generators on full corpus)
- **CI readiness:** BLOCKED (tests pass but production fails)

**Priority:** Fix RandomCandidateGenerator before merge

---

## 📎 APPENDIX

### Test Execution Log
```bash
# pytest suite
$ pytest tests/ -v --tb=short
Result: 22 passed, 7 warnings in 1.77s

# Module validation
$ cd experiments/adaptive_enrichment && python test_modules.py
Result: ✓ ALL TESTS PASSED

# Full corpus generation
$ python generate_test_corpus.py --seed 20260123 --output /tmp/test_corpus.json
Result: Generated 77 semiprimes → /tmp/test_corpus.json

# Full experiment execution
$ python run_experiment.py --corpus /tmp/test_corpus.json --output /tmp/test_results.csv
Result: ValueError: high is out of bounds for int64
```

### Environment
- Python: 3.12.3
- pytest: 9.0.2
- numpy: 2.4.1
- scipy: 1.17.0
- pandas: 3.0.0
- sympy: 1.14.0

### Code Statistics
- Test files: 4 files, 558 lines
- Source files: 8 files, 512 lines
- Documentation: 3 files (CI_INTEGRATION.md, README.md, IMPLEMENTATION_COMPLETE.md)
- Total tests: 22 pytest + 5 module validation = 27 tests

---

**Report Prepared By:** GitHub Copilot  
**Review Requested By:** @zfifteen  
**Date:** 2026-01-24  
**Version:** 1.0
