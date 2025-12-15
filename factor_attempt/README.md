# Blind Factor Attempt for N₁₂₇

This directory contains a single-script implementation for attempting to factor the 127-bit semiprime N₁₂₇ using the existing QMC + resonance + Z5D pipeline.

## Files

- `config_127.py` - Configuration constants for the factorization attempt
- `run_blind_factor_127.py` - Main script that performs the blind factorization attempt

## What This Does

The script attempts to find a non-trivial factor of:

```
N_127 = 137524771864208156028430259349934309717
```

It uses a 30-minute time limit and tests up to 5.2 million candidates from a search window around √N.

## How It Works

1. **QMC Sampling**: Uses Sobol sequences (from `scipy.stats.qmc`) to generate quasi-random candidates in the search range
2. **Resonance Scoring**: Applies geometric resonance scoring (reused from `run_geofac_peaks_mod.py`) to rank candidates
3. **GCD Testing**: Tests the top-K candidates from each batch using `math.gcd()` to find factors
4. **Time Budget**: Stops after 30 minutes or when a non-trivial factor is found

## Running

To run the blind factorization attempt:

```bash
python -m factor_attempt.run_blind_factor_127
```

The script will:
- Display configuration details at startup
- Show progress every 10 batches
- Stop when a factor is found or time expires
- Display total candidates tested and elapsed time

## Output

### If a factor is found:
```
SUCCESS: Non-trivial factor found!
Factor: <factor>
Complement: <complement>
Verification: <factor> * <complement> = <N>
...
```

### If no factor is found:
```
Completed all 80 batches without finding a factor.
Total time: 1800.00s
Total candidates tested: 800,000
```

## Implementation Notes

- **Pure Python integers**: Uses Python's arbitrary-precision integers to avoid overflow
- **Existing code reuse**: All functionality comes from existing modules:
  - QMC generation: `scipy.stats.qmc.Sobol` (as used in `tools/generate_qmc_seeds.py`)
  - Resonance scoring: Geometric phase resonance logic from `tools/run_geofac_peaks_mod.py`
  - Factor testing: Standard `math.gcd()` from Python standard library
- **No new dependencies**: Only uses packages already required by the repository

## Configuration

Key parameters in `config_127.py`:

- `N_127`: The target semiprime (127 bits)
- `SEARCH_MIN`, `SEARCH_MAX`: Search window around √N
- `TOTAL_CANDIDATES`: Number of candidates to test (5.2M)
- `NUM_BATCHES`: Number of batches to process (80)
- `TOP_K_PER_BATCH`: Top candidates per batch to test (10K)
- `MAX_WALLCLOCK_SECONDS`: Time limit (1800s = 30 minutes)

These can be tuned for different performance/coverage tradeoffs.
