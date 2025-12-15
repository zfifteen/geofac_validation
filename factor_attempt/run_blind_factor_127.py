#!/usr/bin/env python3
"""
Blind factorization attempt for N_127 using existing QMC + resonance pipeline.

This script:
1. Uses QMC (Sobol) to generate candidate integers in [SEARCH_MIN, SEARCH_MAX]
2. Scores candidates using geometric resonance (from run_geofac_peaks_mod.py)
3. Tests top-K candidates per batch with GCD
4. Stops after 30 minutes or when a factor is found
"""

import time
import math
import numpy as np
import sys
from pathlib import Path

# Add tools directory to path to import existing functions
tools_path = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(tools_path))

from generate_qmc_seeds import generate_sobol_sequence
from run_geofac_peaks_mod import compute_geometric_resonance

from factor_attempt.config_127 import (
    N_127, SEARCH_MIN, SEARCH_MAX,
    MAX_WALLCLOCK_SECONDS, TOTAL_CANDIDATES,
    NUM_BATCHES, TOP_K_PER_BATCH,
)


def generate_batch(batch_id, batch_size):
    """Use existing QMC code to get batch_size floats in [0,1), then map to ints."""
    # Call existing QMC generator with batch_id as seed
    samples = generate_sobol_sequence(batch_size, dimensions=1, seed=batch_id)
    u = samples[:, 0]  # Extract 1D array
    
    # Map [0,1) to [SEARCH_MIN, SEARCH_MAX]
    width = SEARCH_MAX - SEARCH_MIN
    candidates = np.empty(len(u), dtype=object)
    for i, ui in enumerate(u):
        candidates[i] = SEARCH_MIN + int(ui * width)
    
    return candidates


def score_batch(candidates):
    """Use existing resonance/Z5D function to score each candidate."""
    scores = np.zeros(len(candidates))
    
    for i, d in enumerate(candidates):
        # Skip invalid candidates
        if d <= 1 or d >= N_127:
            scores[i] = 0.0
            continue
        
        # Compute phase parameter from candidate position
        # Use candidate's relationship to sqrt(N) as phase input
        normalized_pos = (d - math.isqrt(N_127)) / N_127
        k_or_phase = (normalized_pos + 1.0) / 2.0  # Map to [0, 1]
        
        # Call existing geometric resonance function
        amplitude, _ = compute_geometric_resonance(N_127, k_or_phase, window_size=1000)
        scores[i] = amplitude
    
    return scores


def try_certify_factor(N, d):
    """
    Check if d is a non-trivial factor of N using GCD.
    
    Returns the factor if found, None otherwise.
    """
    g = math.gcd(N, d)
    if 1 < g < N:
        return g
    return None


def main():
    """
    Main factorization attempt loop.
    
    Process batches of QMC-generated candidates, score them with resonance,
    and test top-K for actual factors.
    """
    print(f"Starting blind factor attempt for N_127")
    print(f"N_127 = {N_127}")
    print(f"Search range: [{SEARCH_MIN}, {SEARCH_MAX}]")
    print(f"Total candidates: {TOTAL_CANDIDATES:,}")
    print(f"Batches: {NUM_BATCHES}")
    print(f"Top-K per batch: {TOP_K_PER_BATCH:,}")
    print(f"Max GCD calls: {NUM_BATCHES * TOP_K_PER_BATCH:,} (batches × top-K)")
    print(f"Max time: {MAX_WALLCLOCK_SECONDS}s ({MAX_WALLCLOCK_SECONDS//60} minutes)")
    print()
    
    t0 = time.monotonic()
    batch_size = TOTAL_CANDIDATES // NUM_BATCHES
    
    total_tested = 0
    
    for batch_id in range(NUM_BATCHES):
        elapsed = time.monotonic() - t0
        if elapsed > MAX_WALLCLOCK_SECONDS:
            print(f"\nTime budget exhausted after {elapsed:.1f}s without finding factor.")
            print(f"Total candidates tested: {total_tested:,}")
            return
        
        # Generate candidates
        candidates = generate_batch(batch_id, batch_size)
        
        # Score all candidates in batch
        scores = score_batch(candidates)
        
        # CRITICAL CONSTRAINT: Select ONLY top-K by score
        # GCD will be called ONLY on these top-K candidates, never on all candidates
        # This enforces a maximum of NUM_BATCHES * TOP_K_PER_BATCH GCD calls
        # Use argpartition for O(n) selection of top-K elements (faster than full sort)
        if TOP_K_PER_BATCH < len(scores):
            idx = np.argpartition(scores, -TOP_K_PER_BATCH)[-TOP_K_PER_BATCH:]
        else:
            idx = np.arange(len(scores))
        
        # Test ONLY the top-K candidates with GCD (not all candidates)
        for i in idx:
            d = candidates[i]
            g = try_certify_factor(N_127, d)
            if g is not None:
                elapsed = time.monotonic() - t0
                print(f"\n{'='*60}")
                print(f"SUCCESS: Non-trivial factor found!")
                print(f"{'='*60}")
                print(f"Factor: {g}")
                print(f"Complement: {N_127 // g}")
                print(f"Verification: {g} * {N_127 // g} = {g * (N_127 // g)}")
                print(f"Batch ID: {batch_id}")
                print(f"Candidate index in batch: {int(i)}")
                print(f"Candidate value: {d}")
                print(f"Resonance score: {scores[i]:.6f}")
                print(f"Time elapsed: {elapsed:.2f}s")
                print(f"Total candidates tested: {total_tested + len(idx):,}")
                return
        
        total_tested += len(idx)
        
        # Progress update every 10 batches
        if (batch_id + 1) % 10 == 0:
            elapsed = time.monotonic() - t0
            progress = (batch_id + 1) / NUM_BATCHES * 100
            print(f"Progress: {batch_id + 1}/{NUM_BATCHES} ({progress:.1f}%), "
                  f"time: {elapsed:.1f}s, tested: {total_tested:,}")
    
    elapsed = time.monotonic() - t0
    print(f"\nCompleted all {NUM_BATCHES} batches without finding a factor.")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total candidates tested: {total_tested:,}")


if __name__ == "__main__":
    main()
