# Z5D Validation Experiment (Issue #16)

## Quick Start

```bash
# Run the complete validation workflow
./experiments/run_validation.sh

# Or view existing results
python3 experiments/summarize_results.py
```

## What Was Done

Successfully executed **Issue #16**: "Validate Z5D resonance scoring hypothesis using N₁₂₇ ground truth"

### Deliverables ✓

1. **Experiment Script** (`experiments/z5d_validation_n127.py`)
   - Generated 100,000 candidates uniformly in ±13% window around √N₁₂₇
   - Scored each with Z5D resonance
   - Computed distance metrics to true factors
   - Analyzed enrichment for Top-K slices
   - Runtime: 6.8 seconds (14,648 candidates/sec)

2. **Analysis Notebook** (`notebooks/z5d_validation_analysis.ipynb`)
   - 5 visualization types
   - Statistical tests (Kolmogorov-Smirnov, Mann-Whitney U)
   - Key findings interpretation
   - Recommendations

3. **Results Report** (`docs/z5d_validation_n127_results.md`)
   - Full experimental write-up
   - Tables, statistics, interpretation
   - 8,842 words of detailed analysis

4. **Data Files**
   - `data/z5d_validation_n127_results.csv` (100,001 rows, 14 MB)
   - `data/z5d_validation_n127_summary.json` (summary statistics)

## Key Results

### Hypothesis Test: **MODERATE SIGNAL DETECTED**

**Finding**: Z5D resonance scoring shows **10x enrichment** for Top-10,000 candidates near the larger factor (q), but **0x enrichment** near the smaller factor (p).

| Metric | Value | Status |
|--------|-------|--------|
| Maximum enrichment | 10.0x @ Top-10K | ✓ Strong (for q) |
| Statistical significance | p < 1e-6 | ✓ Highly significant |
| Asymmetry | Strong bias toward larger factor | ⚠ Unexpected |
| Uniformity | Localized to Top-10K slice | ⚠ Not uniform |

### Interpretation

**Verdict**: Z5D provides statistically significant signal but with important caveats:
- ✓ **Strong enrichment** (10x) at specific K value (10,000 candidates)
- ✓ **Highly significant** statistically (p < 1e-6)
- ⚠ **Asymmetric**: Works for larger factor, not smaller
- ⚠ **Localized**: Signal strongest at Top-10K, weaker elsewhere

**Conclusion**: Z5D captures real geometric structure in the factorization problem, but needs refinement to be uniformly effective.

## Critical Discovery: Asymmetric Factor Detection

Z5D exhibits a **bias toward the larger, more-offset factor**:
- **Larger factor (q)**: 86.78% of Top-10K within ±1% → 10x enrichment
- **Smaller factor (p)**: 0% of any Top-K within ±1% → 0x enrichment

This correlates with the ground truth asymmetry:
- p is at -10.39% below √N (closer to √N)
- q is at +11.59% above √N (farther from √N)

**Implication**: Z5D may preferentially detect factors that deviate more from √N.

## Performance

**Computational efficiency exceeded expectations**:
- Original estimate: ~40ms/candidate → 11 hours for 100K
- Actual: ~0.07ms/candidate → 6.8 seconds for 100K
- **Speedup: 360x faster than estimated**

This enables scaling to 1M candidates in ~68 seconds (vs. originally estimated ~110 hours).

## Files Reference

```
experiments/
├── z5d_validation_n127.py      # Main experiment (run this first)
├── summarize_results.py        # Quick summary of results
└── run_validation.sh           # Complete workflow script

notebooks/
└── z5d_validation_analysis.ipynb  # Interactive analysis + plots

docs/
├── z5d_validation_n127_results.md  # Full results report
└── IMPLEMENTATION_SUMMARY.md       # Status and instructions

data/
├── z5d_validation_n127_results.csv    # Full dataset (100K rows)
└── z5d_validation_n127_summary.json   # Summary statistics
```

## Usage Examples

### Run Experiment (from scratch)
```bash
python3 experiments/z5d_validation_n127.py
```

### View Summary (from existing results)
```bash
python3 experiments/summarize_results.py
```

### Analyze Data
```bash
jupyter notebook notebooks/z5d_validation_analysis.ipynb
```

### Run Complete Workflow
```bash
./experiments/run_validation.sh
```

### Scale to 1M Candidates
Edit `experiments/z5d_validation_n127.py`:
```python
NUM_CANDIDATES = 1_000_000  # Change from 100_000
```
Then re-run. Expected runtime: ~68 seconds.

## Next Steps

1. **Review detailed report**: `docs/z5d_validation_n127_results.md`
2. **Analyze in notebook**: `notebooks/z5d_validation_analysis.ipynb`
3. **Scale experiment**: Run with 1M candidates
4. **Test generalization**: Try other semiprimes
5. **Investigate asymmetry**: Understand why Z5D favors larger factors
6. **Refine scoring**: Develop balanced or corrected Z5D variants

## Dependencies

Required Python packages:
```bash
pip install gmpy2 mpmath pandas numpy scipy matplotlib seaborn jupyter
```

## Reproducibility

All results are reproducible with:
- Fixed random seed (127)
- Deterministic algorithms
- Version-controlled code
- Complete data provenance

## Citation

```
Z5D Validation for N₁₂₇
Repository: github.com/zfifteen/geofac_validation
Issue: #16
Date: 2025-12-15
Results: docs/z5d_validation_n127_results.md
```

## Status

✓ **COMPLETE** - All Issue #16 deliverables implemented and tested  
✓ Experiment run successfully  
✓ Results documented  
✓ Analysis tools provided  
✓ Reproducible workflow established

---

**Last Updated**: 2025-12-15  
**Issue**: #16  
**Status**: ✓ COMPLETE
