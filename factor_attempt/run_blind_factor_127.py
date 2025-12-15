import time
import math
import numpy as np
from scipy.stats import qmc

from factor_attempt.config_127 import (
    N_127,
    SEARCH_MIN,
    SEARCH_MAX,
    MAX_WALLCLOCK_SECONDS,
    TOTAL_CANDIDATES,
    NUM_BATCHES,
    TOP_K_PER_BATCH,
)


def generate_batch(batch_id, batch_size):
    """Use existing QMC code to get batch_size floats in [0,1), then map to ints."""
    sobol = qmc.Sobol(d=1, scramble=True, seed=batch_id)
    u = sobol.random(batch_size).flatten()
    width = SEARCH_MAX - SEARCH_MIN
    d = SEARCH_MIN + (u * width).astype(np.int64)
    return d


def score_candidate(N, d, k_or_phase):
    """Compute resonance score for a single candidate d."""
    phi = (1 + np.sqrt(5)) / 2
    e = np.e
    phase_angle = k_or_phase * 2 * np.pi
    resonance = 0.0
    # Geometric phase resonance with golden ratio
    phase_term = np.cos(phase_angle + np.log(d) * phi)
    resonance += abs(phase_term) * (1.0 / np.log(max(2, d)))
    # E-based harmonic
    e_term = np.cos(np.log(d) * e)
    resonance += abs(e_term) * 0.5
    return resonance


def score_batch(candidates):
    """Use resonance scoring for each candidate."""
    scores = []
    for d in candidates:
        score = score_candidate(N_127, d, k_or_phase=0.0)
        scores.append(score)
    return np.array(scores)


def try_certify_factor(N, d):
    g = math.gcd(N, int(d))
    if 1 < g < N:
        return g
    return None


def main():
    t0 = time.monotonic()
    batch_size = 65536  # Power of 2 for Sobol
    num_batches = min(NUM_BATCHES, TOTAL_CANDIDATES // batch_size)

    for batch_id in range(num_batches):
        if time.monotonic() - t0 > MAX_WALLCLOCK_SECONDS:
            print("Time budget exhausted without factor.")
            return

        candidates = generate_batch(batch_id, batch_size)
        scores = score_batch(candidates)

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
