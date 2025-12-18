#!/usr/bin/env python3
"""
Blind factorization attempt for N_127 using existing geofac_validation pipeline.

This script attempts to find a non-trivial factor of N_127 within a 30-minute
wall-clock limit using only pre-existing QMC and Z5D/resonance scoring modules.

Usage:
    python -m factor_attempt.run_blind_factor_127
"""

import sys
import time
from pathlib import Path

import gmpy2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from factor_attempt.config_127 import (
    N_127,
    SQRT_N,
    SEARCH_MIN,
    SEARCH_MAX,
    MAX_WALLCLOCK_SECONDS,
    TOTAL_CANDIDATES,
    NUM_BATCHES,
    BATCH_SIZE,
    TOP_K_PER_BATCH,
    QMC_DIMENSIONS,
    QMC_SEED,
)
from tools.generate_qmc_seeds import generate_sobol_sequence
from z5d_adapter import z5d_n_est, compute_z5d_score


def map_qmc_to_search_range(qmc_samples: np.ndarray) -> list:
    """
    Map QMC samples from [0,1)^d to candidate odd integers in [SEARCH_MIN, SEARCH_MAX].

    Uses two QMC dimensions to build a 106-bit fixed-point fraction, avoiding float
    quantization at ~1e19 scales (float64 has ~53 bits of precision).
    """
    if qmc_samples.shape[1] < 2:
        raise ValueError("Need at least 2 QMC dimensions for integer-range mapping")

    search_min = int(SEARCH_MIN)
    search_max = int(SEARCH_MAX)
    search_range = search_max - search_min

    scale = 1 << 53
    denom_bits = 106

    candidates = []
    for row in qmc_samples:
        hi = min(int(row[0] * scale), scale - 1)
        lo = min(int(row[1] * scale), scale - 1)

        x = (hi << 53) | lo
        offset = (x * (search_range + 1)) >> denom_bits

        candidate = search_min + offset
        if candidate & 1 == 0:
            candidate += 1
            if candidate > search_max:
                candidate -= 2

        candidates.append(gmpy2.mpz(candidate))

    return candidates


def compute_resonance_score(candidate: gmpy2.mpz, qmc_row: np.ndarray) -> float:
    """
    Compute a resonance score for a candidate factor using Z5D scoring.

    Combines the Z5D score with QMC phase information for ranking candidates.
    """
    try:
        n_est = z5d_n_est(str(candidate))
        z5d_score = compute_z5d_score(str(candidate), n_est)

        phi = (1 + np.sqrt(5)) / 2
        phase = qmc_row[3] if len(qmc_row) > 3 else 0.5
        phase_contribution = abs(np.cos(phase * 2 * np.pi * phi))

        resonance = -z5d_score + phase_contribution * 2.0
        return resonance
    except Exception:
        return -1000.0


def check_candidate(candidate: gmpy2.mpz) -> tuple:
    """
    Check if candidate is a non-trivial factor of N_127 using GCD.

    Returns:
        Tuple of (is_factor, factor_value)
    """
    g = gmpy2.gcd(candidate, N_127)

    if g > 1 and g < N_127:
        return True, g
    return False, None


def run_batch(batch_num: int, start_time: float) -> tuple:
    """
    Run a single batch of candidate generation and testing.

    Returns:
        Tuple of (found_factor, factor_value, candidates_tested)
    """
    elapsed = time.time() - start_time
    if elapsed >= MAX_WALLCLOCK_SECONDS:
        return None, None, 0

    batch_seed = QMC_SEED + batch_num
    qmc_samples = generate_sobol_sequence(BATCH_SIZE, QMC_DIMENSIONS, batch_seed)

    candidates = map_qmc_to_search_range(qmc_samples)

    scored_candidates = []
    for i, candidate in enumerate(candidates):
        if time.time() - start_time >= MAX_WALLCLOCK_SECONDS:
            print(f"  Timeout during scoring at candidate {i}", file=sys.stderr)
            break

        score = compute_resonance_score(candidate, qmc_samples[i])
        scored_candidates.append((candidate, score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_candidates[:TOP_K_PER_BATCH]

    candidates_tested = 0
    for candidate, score in top_candidates:
        if time.time() - start_time >= MAX_WALLCLOCK_SECONDS:
            print(f"  Timeout during GCD testing", file=sys.stderr)
            break

        candidates_tested += 1
        is_factor, factor = check_candidate(candidate)

        if is_factor:
            return True, factor, candidates_tested

    return False, None, candidates_tested


def main():
    """Main entry point for the blind factorization attempt."""
    print("=" * 60)
    print("N_127 Blind Factorization Attempt")
    print("=" * 60)
    print(f"Target: N_127 = {N_127}")
    print(f"Bit length: {gmpy2.bit_length(N_127)}")
    print(f"sqrt(N): {SQRT_N}")
    print(f"Search range: [{SEARCH_MIN}, {SEARCH_MAX}]")
    print(f"Total candidates: {TOTAL_CANDIDATES:,}")
    print(f"Batches: {NUM_BATCHES}")
    print(f"Top-K per batch: {TOP_K_PER_BATCH:,}")
    print(f"Wall-clock limit: {MAX_WALLCLOCK_SECONDS} seconds (30 minutes)")
    print("=" * 60)
    print()

    start_time = time.time()
    total_tested = 0
    factor_found = False
    found_factor = None

    for batch_num in range(NUM_BATCHES):
        elapsed = time.time() - start_time
        if elapsed >= MAX_WALLCLOCK_SECONDS:
            print(f"\nTimeout reached after {elapsed:.1f} seconds")
            break

        remaining = MAX_WALLCLOCK_SECONDS - elapsed
        print(f"Batch {batch_num + 1}/{NUM_BATCHES} "
              f"(elapsed: {elapsed:.1f}s, remaining: {remaining:.1f}s)")

        found, factor, tested = run_batch(batch_num, start_time)
        total_tested += tested

        if found:
            factor_found = True
            found_factor = factor
            print(f"\n{'=' * 60}")
            print("FACTOR FOUND!")
            print(f"{'=' * 60}")
            print(f"Factor: {factor}")
            print(f"Cofactor: {N_127 // factor}")
            print(f"Verification: {factor} * {N_127 // factor} = {factor * (N_127 // factor)}")
            print(f"Match: {factor * (N_127 // factor) == N_127}")
            break

        print(f"  Tested {tested} candidates, no factor found yet")
        print(f"  Total tested so far: {total_tested:,}")

    end_time = time.time()
    total_elapsed = end_time - start_time

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Total candidates tested: {total_tested:,}")
    print(f"Factor found: {factor_found}")

    if factor_found:
        print(f"Factor: {found_factor}")
        print(f"Cofactor: {N_127 // found_factor}")
        return 0
    else:
        print("No non-trivial factor found within time limit.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
