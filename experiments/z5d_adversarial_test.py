#!/usr/bin/env python3
"""
Z5D Validation for any semiprime: Generate candidates and score with Z5D resonance.
"""

import random
import csv
import gmpy2
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from z5d_adapter import z5d_n_est, compute_z5d_score


def validate_semiprime(semiprime, num_candidates=100000, random_seed=42):
    """
    Validate Z5D on a semiprime with known factors.
    Returns summary dict.
    """
    name = semiprime["name"]
    N_str = semiprime["N"]
    P_str = semiprime["p"]
    Q_str = semiprime["q"]

    N = gmpy2.mpz(N_str)
    P = gmpy2.mpz(P_str)
    Q = gmpy2.mpz(Q_str)

    # Compute sqrt(N) using gmpy2
    SQRT_N = gmpy2.isqrt(N)
    SQRT_N_STR = str(SQRT_N)

    # Window: ±13% around sqrt(N)
    WINDOW_PERCENT = 13
    WINDOW_RADIUS = SQRT_N * WINDOW_PERCENT // 100
    SEARCH_MIN = SQRT_N - WINDOW_RADIUS
    SEARCH_MAX = SQRT_N + WINDOW_RADIUS

    print(f"Validating {name}: N has {len(N_str)} digits")
    print(f"Window: [{SEARCH_MIN}, {SEARCH_MAX}]")

    # Set seed
    random.seed(random_seed)

    range_size = int(SEARCH_MAX - SEARCH_MIN)
    candidates = set()

    while len(candidates) < num_candidates:
        offset = random.randint(0, range_size)
        candidate = SEARCH_MIN + offset
        if candidate % 2 == 0:
            candidate += 1
        if candidate > SEARCH_MAX:
            candidate -= 2
        candidates.add(candidate)

    candidates = sorted(list(candidates))

    results = []
    for i, c in enumerate(candidates):
        if i % 10000 == 0:
            print(f"Processed {i}/{num_candidates} candidates")

        c_str = str(c)
        n_est = z5d_n_est(c_str)
        z5d_score = compute_z5d_score(c_str, n_est)

        dist_p = abs(c - P)
        dist_q = abs(c - Q)
        percent_from_sqrt = float((c - SQRT_N) * 100 / SQRT_N)

        results.append(
            {
                "candidate": c_str,
                "z5d_score": z5d_score,
                "distance_to_p": str(dist_p),
                "distance_to_q": str(dist_q),
                "percent_from_sqrt": percent_from_sqrt,
            }
        )

    # Sort by score
    results_sorted = sorted(results, key=lambda x: x["z5d_score"])

    # Compute enrichment for Top-10K and Top-100K
    # Define zone as near p or q: within 5% of their positions
    p_percent = float((P - SQRT_N) * 100 / SQRT_N)
    q_percent = float((Q - SQRT_N) * 100 / SQRT_N)

    # Zone: from min(p,q)-5% to max(p,q)+5%
    zone_min = min(p_percent, q_percent) - 5
    zone_max = max(p_percent, q_percent) + 5

    # Baseline: uniform in window
    window_min = float((SEARCH_MIN - SQRT_N) * 100 / SQRT_N)
    window_max = float((SEARCH_MAX - SQRT_N) * 100 / SQRT_N)
    zone_width = zone_max - zone_min
    window_width = window_max - window_min
    baseline_percent = zone_width / window_width

    def compute_enrichment(top_k):
        top_k_results = results_sorted[:top_k]
        in_zone = sum(
            1 for r in top_k_results if zone_min <= r["percent_from_sqrt"] <= zone_max
        )
        actual_percent = in_zone / top_k
        enrichment = actual_percent / baseline_percent if baseline_percent > 0 else 0
        return enrichment

    enrichment_10k = compute_enrichment(10000)
    enrichment_100k = compute_enrichment(100000)

    # Find detected factor
    top_10k = results_sorted[:10000]
    near_p = sum(1 for r in top_10k if abs(r["percent_from_sqrt"] - p_percent) <= 5)
    near_q = sum(1 for r in top_10k if abs(r["percent_from_sqrt"] - q_percent) <= 5)

    detected = "q" if near_q > near_p else "p"
    offset = q_percent if detected == "q" else p_percent

    summary = {
        "name": name,
        "N": N_str,
        "p": P_str,
        "q": Q_str,
        "sqrt_N": SQRT_N_STR,
        "p_offset_percent": p_percent,
        "q_offset_percent": q_percent,
        "detected_factor": detected,
        "offset_from_sqrt": offset,
        "top_10k_enrichment": enrichment_10k,
        "top_100k_enrichment": enrichment_100k,
        "candidates_tested": num_candidates,
    }

    # Save CSV
    csv_path = f"experiments/z5d_validation_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    with open(csv_path, "w", newline="") as csvfile:
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

    print(f"Saved results to {csv_path}")
    return summary


if __name__ == "__main__":
    # Run synthetics
    from synthetics_data import synthetics

    summaries = []
    for sp in synthetics:
        summary = validate_semiprime(sp)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    # Save summaries
    with open("experiments/synthetics_test_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)
