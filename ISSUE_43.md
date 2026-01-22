# Issue #43: Prospective Gradient Descent Validation Protocol

## Executive Summary

This issue documents the implementation and validation protocol for the **Gradient Descent "Zoom" Algorithm**, designed to transform GeoFac from a statistical distinguisher into an operational factorization tool.

## Background

Following the validation of PR #42 and the identification of the Coverage Paradox, independent research confirms that:

1. ✅ **Z5D scoring functions as an effective fitness landscape** (p < 10⁻³⁰⁰ statistical significance)
2. ✅ **O(1) scaling validated** across 13 orders of magnitude
3. ❌ **Blind QMC sampling is computationally infeasible** (10⁻¹¹% actual coverage vs 10⁹ samples needed for 50% probability)

**Proposed Solution:** Leverage the Z5D "gradient" to iteratively narrow the search window, transforming a near-impossible lottery into a tractable optimization problem.

---

## Gradient Descent Strategy

### Three-Stage Architecture

#### Stage 1: Blind QMC (Current Approach)
- **Status:** ❌ Failed (0/7 success rate)
- **Root Cause:** Sampling density 10⁶× below birthday paradox threshold
- **Outcome:** Strong statistical signal detected, but no direct hits

#### Stage 2: Gradient-Guided Zoom (This Issue)
- **Method:** Iterative window narrowing based on Z5D peak clustering
- **Target:** Reduce search space from 10¹⁸ to 10⁹ (~10⁹× compression)
- **Expected Iterations:** 5-10 (each narrows by 100×)
- **Computational Cost:** 10⁵ candidates per iteration × 10 iterations = 10⁶ total candidates
- **Success Probability:** 50-80% (assuming Z5D gradient maintains direction)

#### Stage 3: Coppersmith Handoff (Future Work)
- **Trigger:** When window size < N^(1/4) ≈ 3.7 × 10⁹
- **Method:** LLL lattice reduction on polynomial with center of window as approximate root
- **Complexity:** Polynomial time (typically < 1 minute for 127-bit)
- **Success Probability:** 95%+ if window correctly centered

---

## Algorithm Specification

### Iterative Narrowing Protocol

```python
def gradient_zoom(N, initial_window_pct=0.13, zoom_factor=100, max_iterations=10):
    """
    Gradient-guided iterative window narrowing.
    
    Args:
        N: Semiprime to factor (gmpy2.mpz)
        initial_window_pct: Initial window as fraction of √N (default: 0.13)
        zoom_factor: Window reduction per iteration (default: 100)
        max_iterations: Maximum zoom iterations (default: 10)
    
    Returns:
        dict: Results including factor (if found), window history, convergence stats
    """
    sqrt_N = gmpy2.isqrt(N)
    window_center = sqrt_N
    window_radius = int(sqrt_N * initial_window_pct)
    
    for iteration in range(max_iterations):
        # Step 1: Survey - Sample candidates across current window
        candidates = generate_qmc_candidates(
            window_center - window_radius,
            window_center + window_radius,
            n_samples=100_000
        )
        
        # Step 2: Score - Evaluate all candidates with Z5D
        scored = score_candidates_z5d(candidates)
        
        # Step 3: Locate - Identify cluster of top 1% scores
        top_candidates = scored[:int(len(scored) * 0.01)]
        cluster_center = compute_cluster_center(top_candidates)
        
        # Step 4: Test - Check if any top candidates are factors
        for candidate, score in top_candidates:
            if gmpy2.gcd(candidate, N) > 1:
                return {"factor_found": True, "factor": candidate, ...}
        
        # Step 5: Zoom - Re-center and shrink window
        window_center = cluster_center
        window_radius = window_radius // zoom_factor
        
        # Step 6: Convergence check
        if window_radius < N^(1/4):
            # Ready for Coppersmith handoff (Stage 3)
            break
    
    return {"factor_found": False, ...}
```

### Logarithmic Convergence Analysis

If each iteration narrows by 100×, achieving 10⁹× reduction (10¹⁸ → 10⁹) requires:

```
Iterations = log(10⁹) / log(100) = 9 / 2 = 4.5 ≈ 5 iterations
```

Each iteration adds **~6.6 bits of precision** (log₂(100) ≈ 6.64).

---

## Prospective Validation Protocol

To establish GeoFac's operational viability and avoid selection bias, implement a **pre-registered prospective study**.

### Design Requirements

#### 1. Dataset Generation

Generate **20 fresh semiprimes** with pre-specified characteristics:

**10 semiprimes in 80-100 bit range:**
- 3 balanced: |p - q| < 0.05√N
- 4 moderate: 0.10√N < |p - q| < 0.30√N
- 3 extreme: 0.50√N < |p - q| < 1.00√N

**10 semiprimes in 120-140 bit range:**
- 2 balanced: |p - q| < 0.05√N
- 3 moderate: 0.10√N < |p - q| < 0.30√N
- 5 extreme: 0.50√N < |p - q| < 1.00√N

**Pre-registration:** Publish N values WITHOUT factors (commit hash as proof of pre-registration)

#### 2. Execution Protocol

- Run Gradient Zoom algorithm with production parameters (10⁵ candidates/iteration)
- Time limit: 24 hours per semiprime on single consumer machine
- Log all window centers, widths, and Z5D score distributions
- Track total candidates tested, iterations, and convergence behavior

#### 3. Success Criteria

| Tier | Success Rate | Confidence Level |
|------|--------------|------------------|
| **Minimum** | ≥3/20 (15%) | Weak evidence |
| **Target** | ≥10/20 (50%) | Strong evidence |
| **Stretch** | ≥15/20 (75%) | Excellent evidence |

#### 4. Comparative Benchmarking

Run same semiprimes through:
- Pollard's Rho
- ECM (GMP-ECM implementation)
- GNFS (CADO-NFS)
- GeoFac Gradient Zoom

**Report:**
- Time-to-factor (wall clock)
- CPU-hours
- Success/failure status
- Memory usage

### Metrics Beyond Binary Success

- **Bits of Precision Gained:** log₂(initial_window / final_window)
- **Convergence Rate:** Window reduction per iteration
- **Z5D Signal Strength:** Distance ratio at each iteration
- **Gradient Stability:** Does gradient direction remain consistent?
- **False Convergence Rate:** Percentage of runs that converge to wrong location

---

## Implementation Checklist

### Phase 1: Core Algorithm (Week 1-2)

- [ ] Create `gradient_zoom.py` module
- [ ] Implement `gradient_zoom()` function with iteration logic
- [ ] Add `compute_cluster_center()` helper (median or weighted centroid)
- [ ] Implement window history tracking and logging
- [ ] Add convergence detection (window < N^(1/4) trigger)
- [ ] Verify no int64 usage (maintain gmpy2 arbitrary precision)

### Phase 2: Testing Infrastructure (Week 2-3)

- [ ] Create `test_gradient_zoom.py` with unit tests
- [ ] Test on N_127 (known ground truth)
- [ ] Test convergence on 5 known semiprimes (various bit lengths)
- [ ] Validate cluster center computation
- [ ] Test edge cases (gradient divergence, flat landscapes)

### Phase 3: Prospective Dataset (Week 3-4)

- [ ] Create `generate_prospective_semiprimes.py`
- [ ] Generate 20 semiprimes with specified properties
- [ ] Pre-register by committing N values (factors in separate encrypted file)
- [ ] Document semiprime characteristics (bit length, factor offset)

### Phase 4: Execution and Analysis (Week 4-6)

- [ ] Create `run_prospective_validation.py` execution script
- [ ] Run gradient zoom on all 20 semiprimes
- [ ] Collect convergence statistics and logs
- [ ] Run comparative benchmarks (Pollard, ECM, GNFS)
- [ ] Generate analysis report with success rates and performance metrics

### Phase 5: Documentation (Week 6-7)

- [ ] Update README with gradient descent strategy
- [ ] Create `GRADIENT_ZOOM_FINDINGS.md` with results
- [ ] Document limitations and failure modes
- [ ] Publish prospective study results (pre-print)

---

## Expected Performance

### Best Case Scenario
- **Iterations:** 5
- **Candidates per iteration:** 100,000
- **Time per iteration:** ~35 seconds (based on 1.75s per 5k candidates × 20)
- **Total time:** ~3 minutes
- **Success probability:** 80%

### Average Case Scenario
- **Iterations:** 8
- **Candidates per iteration:** 100,000
- **Time per iteration:** ~35 seconds
- **Total time:** ~5 minutes
- **Success probability:** 50%

### Worst Case Scenario
- **Iterations:** 10 (max)
- **Gradient diverges** or **converges to wrong location**
- **Total time:** ~6 minutes (then timeout or fallback)
- **Success probability:** <10%

---

## Risk Assessment

### Technical Risks

1. **Gradient Divergence (High Risk)**
   - **Issue:** Z5D peak may not correspond to actual factor location
   - **Mitigation:** Track multiple peaks (top 5%), test all in parallel
   - **Fallback:** Revert to broader window if gradient unstable

2. **Local Maxima Trapping (Medium Risk)**
   - **Issue:** Gradient descent converges to spurious peak
   - **Mitigation:** Random restarts, simulated annealing, multi-scale search
   - **Detection:** Monitor Z5D score improvement rate

3. **Flat Landscape (Medium Risk)**
   - **Issue:** Z5D scores show no clear gradient near factor
   - **Mitigation:** Increase sample density, use larger initial window
   - **Detection:** Low variance in top 1% scores

4. **Computational Cost Escalation (Low Risk)**
   - **Issue:** Iterations exceed budget, time-to-factor > GNFS
   - **Mitigation:** Hard timeout at 10 iterations, 24 hours max
   - **Acceptance:** If unsuccessful, document as limitation

---

## Success Criteria for Issue Closure

This issue is considered **resolved** when:

1. ✅ Gradient zoom algorithm implemented and unit tested
2. ✅ Prospective dataset generated and pre-registered
3. ✅ Execution protocol run on all 20 semiprimes
4. ✅ Results analyzed and documented
5. ✅ Comparative benchmarking completed
6. ✅ Findings published in `GRADIENT_ZOOM_FINDINGS.md`

**Minimum Success Threshold:** ≥3/20 semiprimes factored (15% success rate)

If threshold not met, document as negative result and analyze failure modes for future research.

---

## References

1. **PR #42:** Improved experiment design and coverage paradox analysis
2. **MASTER_FINDINGS.md:** O(1) scaling validation and coverage calculation
3. **Section VII (Research Report):** Gradient descent strategy and prior art
4. **Section VIII (Research Report):** Three-stage architecture specification
5. **Section IX (Research Report):** Prospective validation protocol design

---

## Timeline

- **Week 1-2:** Core algorithm implementation
- **Week 2-3:** Testing infrastructure
- **Week 3-4:** Prospective dataset generation
- **Week 4-6:** Execution and benchmarking
- **Week 6-7:** Analysis and documentation

**Target Completion:** 6-7 weeks from issue creation

---

## Contact

For questions about this validation protocol, refer to:
- Repository maintainer: @zfifteen
- Original research report: Independent Research Verification (January 22, 2026)
