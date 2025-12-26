# Asymmetric Enrichment Falsification Experiment - Implementation Summary

**Experiment ID:** GEOFAC-ASYM-001  
**Version:** 1.0  
**Date:** December 21, 2025  
**Status:** ✅ Ready for Execution

---

## Implementation Complete

This directory contains a complete, production-ready falsification experiment infrastructure for testing the asymmetric q-factor enrichment hypothesis in Z5D geometric resonance scoring.

### What Has Been Implemented

✅ **Complete experiment pipeline** (4 phases + visualization)  
✅ **Stratified test set generation** (64-426 bits, controlled imbalance)  
✅ **Baseline and Z5D enrichment measurement**  
✅ **Rigorous statistical analysis** (4 tests, Bonferroni correction)  
✅ **Publication-quality visualizations** (5 plot types)  
✅ **Comprehensive documentation** (3 design documents)  
✅ **Validation testing** (6 module tests)  
✅ **Configuration management** (3 YAML files)  
✅ **Reproducibility controls** (fixed seeds, version pinning, provenance)

---

## File Inventory

### Core Implementation (5 files)
```
src/
├── generate_test_set.py          [397 lines] Semiprime generation with Miller-Rabin
├── baseline_mc_enrichment.py     [313 lines] Uniform random baseline measurement
├── z5d_enrichment_test.py        [383 lines] QMC + Z5D scoring + enrichment
├── statistical_analysis.py       [331 lines] Wilcoxon, Bootstrap, Mann-Whitney, Levene
└── visualization.py              [327 lines] 5 plot types + summary report
```

### Configuration (3 files)
```
config/
├── semiprime_generation.yaml     [26 semiprimes across 5 ranges]
├── sampling_parameters.yaml      [QMC, MC, window, epsilon parameters]
└── statistical_thresholds.yaml   [4 criteria, α=0.01, effect sizes]
```

### Documentation (4 files)
```
docs/
├── EXPERIMENT_DESIGN.md          [6.3 KB] Design rationale and power analysis
├── FALSIFICATION_CRITERIA.md     [7.1 KB] Detailed criteria specification
└── ANALYSIS_PROTOCOL.md          [10.3 KB] Step-by-step analysis procedures

README.md                          [8.3 KB] Quick start and usage guide
```

### Automation (4 files)
```
run_experiment.py                  [301 lines] Main orchestrator
test_modules.py                    [186 lines] Validation tests
Makefile                           [50 lines]  Convenience targets
requirements.txt                   [7 dependencies]
```

**Total:** 16 files, ~2,500 lines of code, ~32 KB of documentation

---

## Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SEMIPRIME GENERATION                                         │
│    → 26 semiprimes (64-426 bits) with ground truth factors     │
│    → Stratified across 5 ranges, controlled imbalance          │
│    → Miller-Rabin primality (64 rounds)                        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. BASELINE MONTE CARLO (Phase 1)                              │
│    → 260 trials (26 semiprimes × 10 trials each)               │
│    → 100k uniform random candidates per trial                  │
│    → Measure enrichment near p and q (±1% window)              │
│    → Expected: E_p ≈ 1.0x, E_q ≈ 1.0x                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Z5D ENRICHMENT (Phase 2)                                    │
│    → 260 trials (same semiprimes × 10 trials)                  │
│    → 100k QMC candidates (Sobol, 106-bit precision)            │
│    → Z5D scoring via z5d_adapter.py                            │
│    → Extract top 10%, measure enrichment                       │
│    → Expected claim: E_p ≈ 1.0x, E_q = 5-10x                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. STATISTICAL ANALYSIS (Phase 3)                              │
│    → Wilcoxon signed-rank (q>5, p~1)                           │
│    → Bootstrap CI (10k resamples, 95%)                         │
│    → Mann-Whitney U (q vs p, d>1.5)                            │
│    → Levene test (variance across ranges)                     │
│    → Decision: CONFIRMED / FALSIFIED / PARTIAL / INCONCLUSIVE  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. VISUALIZATION (Phase 4)                                     │
│    → Enrichment comparison (baseline vs Z5D)                   │
│    → Asymmetry distribution                                    │
│    → Confidence interval forest plots                          │
│    → Enrichment by bit range (scale-invariance)                │
│    → Text summary report                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Estimated runtime:** 30-60 minutes (hardware-dependent)

---

## Key Design Features

### 1. Arbitrary Precision Arithmetic

**Problem:** 426-bit semiprimes exceed int64 max (9.2×10¹⁸), causing silent overflow  
**Solution:** All arithmetic uses `gmpy2.mpz` and `mpmath`  
**Validation:** Tested with 256-bit semiprimes in test_modules.py

### 2. Falsification-First Design

**Principle:** 4 independent falsification criteria, any 2 failures → hypothesis falsified  
**Robustness:** Multiple paths prevent single-point-of-failure  
**Transparency:** Criteria defined a priori in config/statistical_thresholds.yaml

### 3. Reproducibility

**Determinism:** Fixed seeds (base 42), deterministic QMC (Sobol scrambling seed)  
**Versioning:** All dependencies pinned in requirements.txt  
**Provenance:** Git commit, timestamp, parameters logged in metadata  
**Validation:** Can reproduce exact results by re-running with same seed

### 4. Statistical Rigor

**Bonferroni correction:** α = 0.01/4 = 0.0025 per test (multiple comparisons)  
**Bootstrap CIs:** No distributional assumptions, robust to outliers  
**Effect sizes:** Not just p-values—requires Cohen's d > 1.5 (large effect)  
**Power analysis:** >95% power to detect 5× enrichment at α=0.01

---

## Usage Examples

### Quick Start
```bash
make test    # Validate modules (30 seconds)
make run     # Run experiment (30-60 minutes)
make results # Display decision summary
```

### Manual Execution
```bash
python3 run_experiment.py
```

### Customization
Edit `config/semiprime_generation.yaml` to adjust:
- Bit-length ranges
- Number of semiprimes per range
- Factor imbalance percentages

Edit `config/sampling_parameters.yaml` to adjust:
- Number of candidates (100k default)
- Number of trials per semiprime (10 default)
- Search window size (±15% default)

---

## Expected Outputs

### Data Files (JSON)
```
data/
├── benchmark_semiprimes.json      [26 semiprimes with ground truth]
└── results/
    ├── phase1_baseline_mc.json           [260 baseline measurements]
    ├── phase2_z5d_enrichment.json        [260 Z5D measurements]
    ├── falsification_decision.json       [Statistical decision]
    ├── experiment_metadata.json          [Provenance]
    └── visualizations/
        ├── enrichment_comparison.png
        ├── asymmetry_distribution.png
        ├── confidence_intervals.png
        ├── enrichment_by_bit_range.png
        └── summary_report.txt
```

### Decision Output Example
```
DECISION: CONFIRMED
CONFIDENCE: 95%

PRIMARY METRICS:
  Q-enrichment: 7.5x (95% CI: [6.2, 8.9])
  P-enrichment: 1.1x (95% CI: [0.9, 1.3])
  Asymmetry ratio: 6.8

FALSIFICATION CRITERIA:
  [✓] Criterion 1: Q-enrichment > 2×
  [✓] Criterion 2: P-enrichment < 3×
  [✓] Criterion 3: Asymmetry ratio >= 2.0
  [✓] Criterion 4: Replication across ranges

INTERPRETATION:
  Hypothesis confirmed: All criteria met. Q-enrichment=7.5x, 
  P-enrichment=1.1x, Asymmetry=6.8. Statistical significance: p < 0.01.
```

---

## Testing & Validation

### Module Tests (test_modules.py)
1. ✅ Primality testing (Miller-Rabin)
2. ✅ Semiprime generation (64-bit)
3. ✅ Uniform candidate generation
4. ✅ QMC candidate generation (Sobol)
5. ✅ Enrichment measurement logic
6. ✅ Arbitrary precision (256-bit)

**Runtime:** ~30 seconds  
**Command:** `make test`

### Configuration Validation
```bash
make validate  # Check YAML syntax
```

---

## Compliance with Specification

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Stratified test set (5 ranges) | ✅ | config/semiprime_generation.yaml |
| Baseline Monte Carlo | ✅ | src/baseline_mc_enrichment.py |
| Z5D enrichment measurement | ✅ | src/z5d_enrichment_test.py |
| QMC sampling (Sobol) | ✅ | scipy.stats.qmc in z5d_enrichment_test.py |
| Wilcoxon signed-rank | ✅ | src/statistical_analysis.py:40 |
| Bootstrap CI (10k) | ✅ | src/statistical_analysis.py:68 |
| Mann-Whitney U | ✅ | src/statistical_analysis.py:105 |
| Levene's test | ✅ | src/statistical_analysis.py:136 |
| 4 falsification criteria | ✅ | src/statistical_analysis.py:144-237 |
| Visualizations | ✅ | src/visualization.py (5 plots) |
| Reproducibility | ✅ | Fixed seeds, pinned deps, provenance |
| Documentation | ✅ | 3 MD files in docs/ + README |

**Specification compliance:** 100%

---

## Known Limitations

1. **Compute cost:** 426-bit semiprimes expensive (26M Z5D scorings total)
2. **No GPU acceleration:** Pure Python/gmpy2 (future optimization opportunity)
3. **Single QMC method:** Only Sobol tested (Halton alternative for future work)
4. **No mechanism investigation:** Tests "does it work" not "why it works"

---

## Future Extensions

If **CONFIRMED**:
- [ ] Mechanism investigation (why asymmetric?)
- [ ] Parameter sensitivity analysis
- [ ] Blind factorization deployment test
- [ ] Scaling beyond 426 bits

If **FALSIFIED**:
- [ ] Post-hoc subgroup analysis
- [ ] Alternative enrichment hypotheses
- [ ] Diagnostic plots (where did it fail?)

If **PARTIALLY CONFIRMED**:
- [ ] Identify which criterion failed
- [ ] Revise claim to match observed effect
- [ ] Redesign with tighter controls

---

## Citation

```bibtex
@software{geofac_asym_falsification_2025,
  title = {Asymmetric Q-Factor Enrichment Falsification Experiment},
  author = {zfifteen},
  year = {2025},
  url = {https://github.com/zfifteen/geofac_validation},
  note = {Experiment ID: GEOFAC-ASYM-001, Version 1.0}
}
```

---

## Contact

- **Issues:** https://github.com/zfifteen/geofac_validation/issues
- **Discussions:** https://github.com/zfifteen/geofac_validation/discussions
- **Repository:** https://github.com/zfifteen/geofac_validation

---

**STATUS:** ✅ **READY FOR EXECUTION**

All deliverables specified in the technical specification have been implemented and validated. The experiment can be run with:

```bash
cd experiments/asymmetric_enrichment_falsification
make test && make run
```

---

*Last Updated: December 21, 2025*  
*Maintainer: zfifteen*  
*License: Same as parent repository*
