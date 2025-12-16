# Z5D Validation on N₁₂₇ (Ground Truth Factors)

## Objective

Test the falsifiable hypothesis:

> Do candidates with the best Z5D alignment (as measured by `z5d_adapter.compute_z5d_score`) concentrate closer to the true factors of **N₁₂₇** than uniform random sampling?

This is an empirical check of whether the Z5D scoring function provides actionable *signal* for factor search, using known ground truth.

## Ground Truth

```
N_127 = 137524771864208156028430259349934309717
p     = 10508623501177419659
q     = 13086849276577416863
sqrt(N_127) = 11727095627827384440
```

Relative to `sqrt(N)`:
- `p` is **10.39022930587% below** `sqrt(N)`
- `q` is **11.59497365676% above** `sqrt(N)`

The validation experiment uses a ±13% window around `sqrt(N)` so both factors are in-range.

## Method

Implementation: `experiments/z5d_validation_n127.py`

1. Uniformly sample **odd** candidates in the ±13% window around `sqrt(N_127)`.
2. Compute `z5d_score` for each candidate (lower/more negative = better Z5D alignment).
3. Rank by `z5d_resonance = -z5d_score` (higher = better).
4. For Top-K slices (K = 100, 1k, 10k, 100k), compute:
   - % within ±1% of `p`
   - % within ±1% of `q`
   - % within ±5% of (`p` or `q`)
   - enrichment factor vs uniform baseline
5. Statistical tests (optional, requires SciPy):
   - KS test on signed distance from `sqrt(N)` (Top-1000 vs random)
   - Mann-Whitney U test on nearest-factor distance (Top-1000 vs random)

## Repro / How to Run

Quick run (no CSV, just summary):
```bash
python3 experiments/z5d_validation_n127.py --num-candidates 100000 --out-csv -
```

Full run (1M rows + CSV):
```bash
python3 experiments/z5d_validation_n127.py --num-candidates 1000000 \\
  --processes 8 \\
  --out-csv artifacts/z5d_validation_n127.csv \\
  --summary-json artifacts/z5d_validation_n127_summary.json
```

Notebook for plots:
- `notebooks/z5d_validation_analysis.ipynb`

## Outputs

- CSV: `artifacts/z5d_validation_n127.csv` (optional; large)
- Summary JSON: `artifacts/z5d_validation_n127_summary.json`

## Results

### Reference Run (2025-12-16 UTC)

This run was executed locally with:
- `num_candidates = 1,000,000`
- `seed = 127`
- `chunk_size = 10,000`
- `processes = 1`
- `topk_max = 100,000`
- `bootstrap_trials = 200`
- runtime: ~`58.37s` scoring + post-analysis (total wall time ~`74s`)

Command:
```bash
env PYTHONUNBUFFERED=1 python3 experiments/z5d_validation_n127.py \
  --num-candidates 1000000 \
  --out-csv - \
  --summary-json artifacts/z5d_validation_n127_summary.json
```

### Baseline Expectation (Uniform Sampling)

Expected fraction (uniform over **odd** candidates in the ±13% window):
- within ±1% of `p`: **0.0689306** (~6.893%)
- within ±1% of `q`: **0.0858423** (~8.584%)
- within ±5% of (`p` or `q`): **0.5413475** (~54.135%)

Empirical baseline from the 1,000,000 sampled candidates matched closely:
- within ±1% of `p`: **0.068825**
- within ±1% of `q`: **0.086035**
- within ±5% of (`p` or `q`): **0.540752**

### Top-K Enrichment (Rank by `z5d_resonance = -z5d_score`)

Fractions are “% of Top-K in the target zone”; enrichment is vs the expected baseline above.

| Top-K | within ±1% of p | enrich | within ±1% of q | enrich | within ±5% of (p or q) | enrich |
|------:|----------------:|-------:|----------------:|-------:|------------------------:|-------:|
| 100 | 0.0000 | 0.00× | 0.0000 | 0.00× | 1.0000 | 1.85× |
| 1,000 | 0.0000 | 0.00× | 0.0000 | 0.00× | 1.0000 | 1.85× |
| 10,000 | 0.0000 | 0.00× | 0.0000 | 0.00× | 1.0000 | 1.85× |
| 100,000 | 0.0000 | 0.00× | 0.86035 | 10.02× | 1.0000 | 1.85× |

Bootstrap 95% CI (enrichment vs expected baseline):
- Top-100,000 within ±1% of `q`: **[9.9996×, 10.0462×]**

### Statistical Tests (Top-1000 vs Random)

With SciPy enabled:
- KS test on signed distance from `sqrt(N)` (Top-1000 vs random): statistic **0.999**, p-value **~0** (underflow to 0.0)
- Mann–Whitney U test on nearest-factor distance (Top-1000 vs random, alternative “Top-1000 is closer”): p-value **3.09e-128**

### Interpretation

- There is strong evidence that high Z5D alignment is **not** spatially uniform (KS statistic ~1).
- In this run, Z5D-aligned candidates show a pronounced **asymmetry**: no enrichment near `p` (within ±1%), and very strong enrichment near `q` (within ±1%) appears only when expanding to Top-100,000.
- Using the issue’s “strong signal” criterion (>5× enrichment for Top-1000), this run is **not** “strong” for the ±1% zones, but it does show **clear signal** on the broader ±5% union zone and via the nearest-factor distance test.

See `notebooks/z5d_validation_analysis.ipynb` for distribution plots (score vs distance, density maps, scatter vs nearest-factor distance).

## Notes / Correctness

This repo forbids fixed-width 64-bit integer types (e.g., `np.int64`, `np.uint64`) because candidate values near `1e19` can silently overflow them. The experiment code uses Python `int` / `gmpy2.mpz` only.
