#!/usr/bin/env python3
"""
Z5D Validation for N₁₂₇: Generate candidates and score with Z5D resonance.

Tests hypothesis: Does Z5D resonance scoring concentrate candidates near true factors better than random?
"""

import random
import csv
import gmpy2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from z5d_adapter import z5d_n_est, compute_z5d_score

# Ground truth from PR #6
N_127_STR = "137524771864208156028430259349934309717"
P_STR = "10508623501177419659"
Q_STR = "13086849276577416863"
SQRT_N_STR = "11727095627827384440"

N_127 = gmpy2.mpz(N_127_STR)
P = gmpy2.mpz(P_STR)
Q = gmpy2.mpz(Q_STR)
SQRT_N = gmpy2.mpz(SQRT_N_STR)

# Window: ±13% around sqrt(N)
WINDOW_PERCENT = 13
WINDOW_RADIUS = SQRT_N * WINDOW_PERCENT // 100
SEARCH_MIN = SQRT_N - WINDOW_RADIUS
SEARCH_MAX = SQRT_N + WINDOW_RADIUS

# Number of candidates
NUM_CANDIDATES = 1_000_000

# Random seed for reproducibility
RANDOM_SEED = 42


def generate_candidates():
    """Generate NUM_CANDIDATES odd numbers uniformly in [SEARCH_MIN, SEARCH_MAX]"""
    random.seed(RANDOM_SEED)

    range_size = int(SEARCH_MAX - SEARCH_MIN)
    candidates = set()

    while len(candidates) < NUM_CANDIDATES:
        # Uniform random in range
        offset = random.randint(0, range_size)
        candidate = SEARCH_MIN + offset

        # Make odd
        if candidate % 2 == 0:
            candidate += 1

        # Ensure within bounds
        if candidate > SEARCH_MAX:
            candidate -= 2

        candidates.add(candidate)

    return sorted(list(candidates))


def compute_distances(candidate):
    """Compute distances to p and q"""
    c = gmpy2.mpz(str(candidate))
    dist_p = abs(c - P)
    dist_q = abs(c - Q)
    return int(dist_p), int(dist_q)


def compute_percent_from_sqrt(candidate):
    """Compute percentage deviation from sqrt(N)"""
    c = gmpy2.mpz(str(candidate))
    percent = (c - SQRT_N) * 100 // SQRT_N
    return float(percent)


def main():
    print(f"Generating {NUM_CANDIDATES} candidates in [{SEARCH_MIN}, {SEARCH_MAX}]...")
    candidates = generate_candidates()

    print("Scoring candidates with Z5D...")
    results = []

    for i, c in enumerate(candidates):
        if i % 10000 == 0:
            print(f"Processed {i}/{NUM_CANDIDATES} candidates")

        c_str = str(c)

        # Compute Z5D score
        n_est = z5d_n_est(c_str)
        z5d_score = compute_z5d_score(c_str, n_est)

        # Compute distances and percent
        dist_p, dist_q = compute_distances(c)
        percent_from_sqrt = compute_percent_from_sqrt(c)

        results.append(
            {
                "candidate": c_str,
                "z5d_score": z5d_score,
                "distance_to_p": dist_p,
                "distance_to_q": dist_q,
                "percent_from_sqrt": percent_from_sqrt,
            }
        )

    print("Saving results to CSV...")
    with open(
        "experiments/z5d_validation_n127_results.csv", "w", newline=""
    ) as csvfile:
        fieldnames = [
            "candidate",
            "z5d_score",
            "distance_to_p",
            "distance_to_q",
            "percent_from_sqrt",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Done!")


if __name__ == "__main__":
    main()
