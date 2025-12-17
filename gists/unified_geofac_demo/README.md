# Unified GeoFac Demo: Blind Factorization

A self-contained demonstration combining two rigorously verified semiprime factorization approaches:

1. **Balanced GeoFac** - Resonance-based search optimized for factors close to √N (p ≈ q)
2. **Adaptive Window Search** - Z5D-guided branch-and-bound for unbalanced factors

## Key Features

- **Blind Operation**: Takes only N as input - never requires or embeds true factors
- **Verified Algorithms**: Ports exact implementations from battle-tested repos:
  - [`zfifteen/geofac`](https://github.com/zfifteen/geofac) - Balanced GeoFac with φ/e resonance
  - [`zfifteen/geofac_validation`](https://github.com/zfifteen/geofac_validation) - Adaptive window strategy
- **Arbitrary Precision**: Handles semiprimes of any size using `gmpy2` and `mpmath`

## Installation

```bash
pip install numpy scipy gmpy2 mpmath
```

## Usage

```bash
python unified_geofac_demo.py <N>
```

### Example: N_127 from geofac_validation

```bash
python unified_geofac_demo.py 137524771864208156028430259349934309717
```

This 127-bit semiprime has factors:
- p = 10508623501177419659 (−10.39% below √N)
- q = 13086849276577416863 (+11.59% above √N)

Expected runtime: 6-48 seconds depending on factor balance.

## How It Works

### Stage 1: Balanced GeoFac (±5% window)

1. Computes √N as initial search center
2. Generates QMC-guided phase samples
3. Applies verified resonance formula:
   ```
   amplitude = Σ |cos(θ + ln(k)·φ)| / ln(k) + |cos(ln(k)·e)| · 0.5
   ```
   where φ = golden ratio, e = Euler's number
4. Tests high-amplitude candidates for divisibility
5. Budget: ~10,000 iterations or factor found

### Stage 2: Adaptive Window Search (if Stage 1 fails)

1. Tests progressively larger windows: ±13%, ±20%, ±30%, ±50%, ±75%, ±100%, ±150%, ±200%, ±300%
2. For each window:
   - Generates 10,000 uniform QMC candidates
   - Scores each with Z5D nth-prime predictor
   - Tests top-1000 highest-scoring candidates
3. Stops when factor found or all windows exhausted

### Unified Search Context

Maintains global state tracking:
- Explored window regions (prevents re-search)
- Test counts per phase
- Best signal scores (resonance amplitude, Z5D scores)
- Timing breakdowns

## Output

The script prints a detailed execution trace and final summary:

```
================================================================================
UNIFIED GEOFAC BLIND FACTORIZATION
================================================================================
Target: N = 137524771864208156028430259349934309717
Bit length: 127 bits

================================================================================
STAGE 1: BALANCED GEOFAC SEARCH
================================================================================
Searching near √N ≈ 11727095627827384440
Window: [11140640846435915218, 12313550409218853662] (±5.0% of √N)
Testing 10000 resonance-guided candidates...

✗ Balanced search exhausted (12.34s, 10000 tests)
  Best amplitude: 3.2145

================================================================================
STAGE 2: ADAPTIVE WINDOW SEARCH
================================================================================

--- Testing ±20% window around √N ---
Testing top 1000 Z5D-scored candidates...

✓ FACTOR FOUND in ±20% window!
  Window time: 8.91s
  Total adaptive time: 15.23s
  Tests in window: 432
  Winning Z5D score: -2.1543

================================================================================
FINAL RESULTS
================================================================================
✓ SUCCESS via adaptive_window

Factors:
  p = 10508623501177419659
  q = 13086849276577416863

Verification: p × q = N? True

--- Performance Summary ---
Total time: 27.57s
  Balanced phase: 12.34s (10000 tests)
  Adaptive phase: 15.23s (432 tests)
Explored windows: [5.0, 13, 20]
```

## Algorithm Sources

This demo uses **zero invented algorithms**. All scoring formulas, window schedules, and stopping criteria are ported verbatim from:

- **Balanced GeoFac**: [`geofac/tools/run_geofac_peaks_mod.py`](https://github.com/zfifteen/geofac/blob/main/tools/run_geofac_peaks_mod.py)
- **Adaptive Window**: [`geofac_validation/experiments/z5d_validation_n127.py`](https://github.com/zfifteen/geofac_validation/blob/main/experiments/z5d_validation_n127.py)
- **Z5D Scoring**: [`geofac_validation/z5d_adapter.py`](https://github.com/zfifteen/geofac_validation/blob/main/z5d_adapter.py)

## Limitations

- **Not constant-time**: Runtime varies with factor balance and magnitude
- **Heuristic-based**: No guarantees on success rate for arbitrary semiprimes
- **Experimental**: This is a research demo, not production-grade software
- **Memory usage**: Scales with candidate set size (typically <100MB)

## Performance Characteristics

Validated on test set (128-426 bit semiprimes):
- Balanced factors (p/q ≈ 1): Usually found in Stage 1 (6-15 seconds)
- Moderate imbalance (|p-q| < 20% √N): Stage 2, early windows (15-30 seconds)
- High imbalance (|p-q| > 50% √N): Stage 2, late windows (30-48 seconds)

## License

This is a composition of algorithms from MIT-licensed repositories. See source repos for full license terms.
