# Unified GeoFac Demo

A self-contained demonstration of blind semiprime factorization using combined balanced and adaptive GeoFac engines.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python unified_geofac_demo.py N
```

Where `N` is the semiprime to factor (as a string for large integers).

## Example

```bash
python unified_geofac_demo.py 137524771864208156028430259349934309717
```

This should factor N_127 from the RSA challenge (if within the demo's limits).

## Description

This script implements a unified factorization approach that works "in the blind" (no access to true factors p,q):

1. **Balanced Phase**: Scans a window around √N for divisibility using geometric resonance heuristics from the original GeoFac implementation.

2. **Adaptive Phase**: If balanced fails, iteratively tests expanding windows [±13%, ±20%, ±30%, ±50%, ±75%, ±100%, ±150%, ±200%, ±300%] around √N. For each window, generates 10,000 Z5D-scored uniform random candidates, sorts by score (lower = better alignment with prime number theorem), and tests the top 100 for divisibility. Terminates immediately upon finding a factor or after exhausting all windows.

## Output

On success:
- Factor pair
- Verification
- Phase statistics (candidates scanned/generated, times, best Z5D score)

On failure:
- Phase statistics

## Limitations

- Demonstration script; not production-optimized for large semiprimes
- Fixed candidate count per window (10,000) and window schedule [±13% to ±300%]
- No parallel processing or dynamic window adaptation
- May require parameter tuning for extreme scales or very unbalanced factors
