"""
Blind 30-minute factor attempt for N_127 using existing QMC + resonance code.

How to run:
    python -m factor_attempt.run_blind_factor_127
"""

import math
import time
from typing import Tuple

import numpy as np

from factor_attempt.config_127 import (
    N_127,
    SEARCH_MIN,
    SEARCH_MAX,
    MAX_WALLCLOCK_SECONDS,
    TOTAL_CANDIDATES,
    NUM_BATCHES,
    TOP_K_PER_BATCH,
)
from tools.generate_qmc_seeds import generate_sobol_sequence
from tools.run_geofac_peaks_mod import compute_geometric_resonance


def generate_batch(batch_id: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate integer candidates and associated phase values.

    Uses the existing Sobol QMC generator (scrambled) with a batch-specific seed
    to keep batches independent while remaining reproducible.
    """
    # Dimensions align with existing pipeline: dim0 -> magnitude, dim3 -> phase
    samples = generate_sobol_sequence(
        num_samples=batch_size, dimensions=4, seed=batch_id + 1
    )

    u = samples[:, 0]
    phases = samples[:, 3]

    width = SEARCH_MAX - SEARCH_MIN
    d = SEARCH_MIN + (u * width).astype(np.int64)
    return d, phases


def _large_n_resonance_scores(phases: np.ndarray) -> np.ndarray:
    """
    Vectorized replica of compute_geometric_resonance's large-N heuristic branch.

    For N_127, sqrt(N) >> 1e6, so the resonance reduces to phase-only terms:
        resonance = |cos(theta)| * 5 + |sin(theta * phi)| * 3
    with theta = phase * 2π and phi = golden ratio.
    """
    phi_const = (1.0 + np.sqrt(5.0)) / 2.0
    theta = phases * (2.0 * np.pi)
    resonance = np.abs(np.cos(theta)) * 5.0 + np.abs(np.sin(theta * phi_const)) * 3.0
    # Match compute_geometric_resonance safeguard: ensure amplitude >= 2.0
    return np.maximum(resonance, 2.0)


def score_batch(phases: np.ndarray) -> np.ndarray:
    """
    Score candidates via existing resonance heuristic.

    Falls back to the full compute_geometric_resonance path for small sqrt(N)
    (not the case here) to stay aligned with the pipeline semantics.
    """
    # Fast path for large N (our case)
    if math.isqrt(N_127) > 10**6:
        return _large_n_resonance_scores(phases)

    # Rare path: small N → use the exact helper per sample
    scores = np.fromiter(
        (compute_geometric_resonance(N_127, float(p))[0] for p in phases),
        dtype=float,
        count=len(phases),
    )
    return scores


def try_certify_factor(N: int, d: int):
    g = math.gcd(N, int(d))
    if 1 < g < N:
        return g
    return None


def main():
    t0 = time.monotonic()
    batch_size = TOTAL_CANDIDATES // NUM_BATCHES

    for batch_id in range(NUM_BATCHES):
        elapsed = time.monotonic() - t0
        if elapsed > MAX_WALLCLOCK_SECONDS:
            print("Time budget exhausted without factor.")
            return

        candidates, phases = generate_batch(batch_id, batch_size)
        scores = score_batch(phases)

        # pick indices of top-K scores
        if TOP_K_PER_BATCH < len(scores):
            idx = np.argpartition(scores, -TOP_K_PER_BATCH)[-TOP_K_PER_BATCH:]
        else:
            idx = np.arange(len(scores))

        for i in idx:
            d = candidates[i]
            g = try_certify_factor(N_127, d)
            if g is not None:
                print(f"SUCCESS: factor found: {g}")
                print(f"batch={batch_id}, candidate_index={int(i)}")
                return

    print("Completed all batches without finding a factor.")


if __name__ == "__main__":
    main()
