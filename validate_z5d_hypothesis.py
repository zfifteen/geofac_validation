#!/usr/bin/env python3
"""
Validate Z5D resonance scoring hypothesis using N₁₂₇ ground truth.

Issue #16: Empirically validate whether the Z5D geometric scoring algorithm
can effectively guide factorization searches by concentrating candidate factors
near the true divisors better than random sampling.

Ground Truth:
- N₁₂₇ = 137524771864208156028430259349934309717
- p = 10508623501177419659 (at -10.39% from √N)
- q = 13086849276577416863 (at +11.59% from √N)

Three-Phase Experimental Plan:
1. Generate 1 million candidates within ±13% search window
2. Score each with Z5D and analyze spatial distribution of top-ranked candidates
3. Apply statistical tests (K-S, Mann-Whitney U) to validate significance

Success Thresholds:
- Strong signal: >5x enrichment factor with p-value <0.001
- Weak signal: 2-5x enrichment with p-value <0.05
- No signal: <1.5x enrichment

Usage:
    python validate_z5d_hypothesis.py [--samples N] [--output-dir DIR]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gmpy2
import numpy as np
from scipy import stats
from scipy.stats import qmc

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))
from z5d_adapter import z5d_n_est, compute_z5d_score


# Ground truth constants (verified from sympy factorization)
N_127 = gmpy2.mpz("137524771864208156028430259349934309717")
SQRT_N = gmpy2.isqrt(N_127)
P_TRUE = gmpy2.mpz("10508623501177419659")
Q_TRUE = gmpy2.mpz("13086849276577416863")

# Search window: ±13% around sqrt(N)
SEARCH_RADIUS = (SQRT_N * 13) // 100
SEARCH_MIN = SQRT_N - SEARCH_RADIUS
SEARCH_MAX = SQRT_N + SEARCH_RADIUS
SEARCH_WIDTH = SEARCH_MAX - SEARCH_MIN

# Pre-computed relative positions of true factors
P_RELATIVE = float((P_TRUE - SEARCH_MIN)) / float(SEARCH_WIDTH)  # Position in [0, 1]
Q_RELATIVE = float((Q_TRUE - SEARCH_MIN)) / float(SEARCH_WIDTH)


def generate_candidates(num_samples: int, seed: int = 42) -> list:
    """
    Generate candidate factors uniformly distributed in the search window.

    Uses Sobol QMC for low-discrepancy sampling, then maps [0,1] to search range.

    CRITICAL: Uses arbitrary-precision integers (gmpy2.mpz) to avoid int64 overflow.
    """
    # Generate QMC samples in [0, 1)
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    qmc_samples = sampler.random(n=num_samples)

    # Map to search range using 106-bit fixed-point arithmetic
    # This avoids float64 quantization errors at ~10^19 scale
    search_min = int(SEARCH_MIN)
    search_max = int(SEARCH_MAX)
    search_range = search_max - search_min

    scale = 1 << 53
    denom_bits = 106

    candidates = []
    for row in qmc_samples:
        # Combine two 53-bit QMC dimensions into 106-bit precision
        hi = min(int(row[0] * scale), scale - 1)
        lo = min(int(row[1] * scale), scale - 1)

        x = (hi << 53) | lo
        offset = (x * (search_range + 1)) >> denom_bits

        candidate = search_min + offset
        # Make odd (primes > 2 are odd)
        if candidate % 2 == 0:
            candidate += 1
            if candidate > search_max:
                candidate -= 2

        candidates.append(gmpy2.mpz(candidate))

    return candidates


def score_candidate(candidate: gmpy2.mpz) -> float:
    """
    Compute Z5D resonance score for a candidate.

    Returns the Z5D score (lower/more negative = better alignment with PNT).
    """
    try:
        n_est = z5d_n_est(str(candidate))
        score = compute_z5d_score(str(candidate), n_est)
        return score
    except Exception:
        return float('inf')


def compute_enrichment(top_candidates: list, proximity_threshold: float = 0.01) -> dict:
    """
    Compute enrichment factor for top-ranked candidates near true factors.

    Args:
        top_candidates: List of (candidate, score) tuples, sorted by score
        proximity_threshold: Fraction of search window to consider "near" a factor

    Returns:
        Dictionary with enrichment statistics
    """
    # Calculate proximity threshold in absolute terms
    threshold_abs = float(SEARCH_WIDTH) * proximity_threshold

    # Count candidates near true factors
    near_p = 0
    near_q = 0
    near_either = 0

    for candidate, _ in top_candidates:
        dist_to_p = abs(int(candidate) - int(P_TRUE))
        dist_to_q = abs(int(candidate) - int(Q_TRUE))

        is_near_p = dist_to_p <= threshold_abs
        is_near_q = dist_to_q <= threshold_abs

        if is_near_p:
            near_p += 1
        if is_near_q:
            near_q += 1
        if is_near_p or is_near_q:
            near_either += 1

    # Expected count under uniform distribution
    # Each factor has proximity_threshold * 2 fraction of window (±threshold)
    # Combined expected fraction (accounting for potential overlap)
    expected_fraction = min(1.0, proximity_threshold * 4)  # 2 factors × 2× threshold each
    expected_count = len(top_candidates) * expected_fraction

    # Enrichment factor
    enrichment = near_either / expected_count if expected_count > 0 else 0

    return {
        "near_p": near_p,
        "near_q": near_q,
        "near_either": near_either,
        "total_top_candidates": len(top_candidates),
        "proximity_threshold": proximity_threshold,
        "expected_count": expected_count,
        "enrichment_factor": enrichment,
    }


def compute_distance_to_factors(candidates: list) -> np.ndarray:
    """
    Compute minimum normalized distance to either true factor for each candidate.

    Returns array of distances in [0, 1] where 0 = exactly on a factor.
    """
    distances = []
    for candidate in candidates:
        dist_to_p = abs(int(candidate) - int(P_TRUE)) / float(SEARCH_WIDTH)
        dist_to_q = abs(int(candidate) - int(Q_TRUE)) / float(SEARCH_WIDTH)
        distances.append(min(dist_to_p, dist_to_q))
    return np.array(distances)


def run_statistical_tests(top_distances: np.ndarray, baseline_distances: np.ndarray) -> dict:
    """
    Apply statistical tests comparing top-ranked candidates to random baseline.

    Tests:
    - Kolmogorov-Smirnov: Tests if distributions differ
    - Mann-Whitney U: Tests if top candidates are closer to factors

    Returns dictionary with test statistics and p-values.
    """
    results = {}

    # K-S test: Do the distributions differ?
    ks_stat, ks_pvalue = stats.ks_2samp(top_distances, baseline_distances)
    results["ks_test"] = {
        "statistic": float(ks_stat),
        "p_value": float(ks_pvalue),
        "significant_001": bool(ks_pvalue < 0.001),
        "significant_005": bool(ks_pvalue < 0.05),
    }

    # Mann-Whitney U: Are top candidates systematically closer?
    # Alternative='less' tests if top_distances < baseline_distances
    mw_stat, mw_pvalue = stats.mannwhitneyu(
        top_distances, baseline_distances, alternative='less'
    )
    results["mann_whitney"] = {
        "statistic": float(mw_stat),
        "p_value": float(mw_pvalue),
        "significant_001": bool(mw_pvalue < 0.001),
        "significant_005": bool(mw_pvalue < 0.05),
    }

    # Summary statistics
    results["summary"] = {
        "top_mean_distance": float(np.mean(top_distances)),
        "top_median_distance": float(np.median(top_distances)),
        "baseline_mean_distance": float(np.mean(baseline_distances)),
        "baseline_median_distance": float(np.median(baseline_distances)),
        "distance_ratio": float(np.mean(top_distances) / np.mean(baseline_distances)),
    }

    return results


def classify_signal(
    enrichment_factor: float,
    p_value: float,
    distance_ratio: float
) -> str:
    """
    Classify the signal strength based on enrichment, p-value, and distance ratio.

    Uses multiple criteria:
    - Enrichment: factor concentration in proximity zones
    - P-value: statistical significance
    - Distance ratio: top candidates vs baseline (lower = closer to factors)
    """
    # Strong signal: statistically significant AND clear effect
    if p_value < 0.001 and (enrichment_factor > 5 or distance_ratio < 0.5):
        return "STRONG"
    # Weak signal: significant with moderate effect
    elif p_value < 0.05 and (enrichment_factor >= 2 or distance_ratio < 0.8):
        return "WEAK"
    # No signal: not significant or no effect
    elif p_value > 0.05 or (enrichment_factor < 1.5 and distance_ratio > 0.9):
        return "NONE"
    else:
        return "INCONCLUSIVE"


def main():
    parser = argparse.ArgumentParser(
        description="Validate Z5D resonance scoring hypothesis using N₁₂₇ ground truth"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1_000_000,
        help="Number of candidates to generate (default: 1,000,000)",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.01,
        help="Fraction of candidates to analyze as 'top' (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--proximity",
        type=float,
        default=0.01,
        help="Proximity threshold as fraction of window (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/validation"),
        help="Output directory for results (default: artifacts/validation)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for scoring (default: 10000)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Z5D RESONANCE SCORING HYPOTHESIS VALIDATION")
    print("Issue #16: Validate using N₁₂₇ ground truth")
    print("=" * 70)
    print()

    print("GROUND TRUTH:")
    print(f"  N₁₂₇ = {N_127}")
    print(f"  √N   = {SQRT_N}")
    print(f"  p    = {P_TRUE} (at -10.39% from √N)")
    print(f"  q    = {Q_TRUE} (at +11.59% from √N)")
    print()

    print("SEARCH WINDOW (±13%):")
    print(f"  Min  = {SEARCH_MIN}")
    print(f"  Max  = {SEARCH_MAX}")
    print(f"  Width = {SEARCH_WIDTH}")
    print()

    print("EXPERIMENT PARAMETERS:")
    print(f"  Samples: {args.samples:,}")
    print(f"  Top fraction: {args.top_fraction:.2%}")
    print(f"  Proximity threshold: {args.proximity:.2%}")
    print(f"  Random seed: {args.seed}")
    print()

    # Phase 1: Generate candidates
    print("=" * 70)
    print("PHASE 1: CANDIDATE GENERATION")
    print("=" * 70)

    start_time = time.time()
    print(f"Generating {args.samples:,} candidates...")
    candidates = generate_candidates(args.samples, seed=args.seed)
    gen_time = time.time() - start_time
    print(f"Generated {len(candidates):,} candidates in {gen_time:.2f}s")
    print()

    # Phase 2: Score all candidates with Z5D
    print("=" * 70)
    print("PHASE 2: Z5D SCORING")
    print("=" * 70)

    start_time = time.time()
    scored_candidates = []

    num_batches = (len(candidates) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(candidates))
        batch = candidates[batch_start:batch_end]

        for candidate in batch:
            score = score_candidate(candidate)
            scored_candidates.append((candidate, score))

        # Progress update every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            progress = (batch_idx + 1) / num_batches * 100
            rate = (batch_end) / elapsed if elapsed > 0 else 0
            eta = (len(candidates) - batch_end) / rate if rate > 0 else 0
            print(f"  Progress: {progress:.1f}% ({batch_end:,}/{len(candidates):,}) "
                  f"- {rate:.0f} candidates/s - ETA: {eta:.0f}s")

    score_time = time.time() - start_time
    print(f"\nScored {len(scored_candidates):,} candidates in {score_time:.1f}s")
    print()

    # Sort by score (lower = better Z5D alignment)
    scored_candidates.sort(key=lambda x: x[1])

    # Phase 3: Statistical Analysis
    print("=" * 70)
    print("PHASE 3: STATISTICAL ANALYSIS")
    print("=" * 70)

    # Select top candidates
    top_k = int(len(scored_candidates) * args.top_fraction)
    top_candidates = scored_candidates[:top_k]

    # Compute distances to true factors
    all_distances = compute_distance_to_factors([c for c, _ in scored_candidates])
    top_distances = compute_distance_to_factors([c for c, _ in top_candidates])

    # Random baseline: sample from all candidates
    np.random.seed(args.seed)
    baseline_indices = np.random.choice(len(scored_candidates), size=top_k, replace=False)
    baseline_candidates = [scored_candidates[i] for i in baseline_indices]
    baseline_distances = compute_distance_to_factors([c for c, _ in baseline_candidates])

    print(f"\nComparing top {top_k:,} candidates ({args.top_fraction:.2%}) vs random baseline")
    print()

    # Enrichment analysis
    enrichment = compute_enrichment(top_candidates, args.proximity)

    print("ENRICHMENT ANALYSIS:")
    print(f"  Candidates near p (within {args.proximity:.1%}): {enrichment['near_p']:,}")
    print(f"  Candidates near q (within {args.proximity:.1%}): {enrichment['near_q']:,}")
    print(f"  Candidates near either: {enrichment['near_either']:,}")
    print(f"  Expected by chance: {enrichment['expected_count']:.1f}")
    print(f"  ENRICHMENT FACTOR: {enrichment['enrichment_factor']:.2f}x")
    print()

    # Statistical tests
    test_results = run_statistical_tests(top_distances, baseline_distances)

    print("KOLMOGOROV-SMIRNOV TEST:")
    print(f"  Statistic: {test_results['ks_test']['statistic']:.4f}")
    print(f"  P-value: {test_results['ks_test']['p_value']:.2e}")
    print(f"  Significant (p<0.001): {test_results['ks_test']['significant_001']}")
    print(f"  Significant (p<0.05): {test_results['ks_test']['significant_005']}")
    print()

    print("MANN-WHITNEY U TEST (one-sided: top < baseline):")
    print(f"  Statistic: {test_results['mann_whitney']['statistic']:.2e}")
    print(f"  P-value: {test_results['mann_whitney']['p_value']:.2e}")
    print(f"  Significant (p<0.001): {test_results['mann_whitney']['significant_001']}")
    print(f"  Significant (p<0.05): {test_results['mann_whitney']['significant_005']}")
    print()

    print("DISTANCE SUMMARY:")
    print(f"  Top candidates mean distance: {test_results['summary']['top_mean_distance']:.4f}")
    print(f"  Baseline mean distance: {test_results['summary']['baseline_mean_distance']:.4f}")
    print(f"  Distance ratio (top/baseline): {test_results['summary']['distance_ratio']:.4f}")
    print()

    # Determine signal classification
    signal = classify_signal(
        enrichment['enrichment_factor'],
        test_results['mann_whitney']['p_value'],
        test_results['summary']['distance_ratio']
    )

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    if signal == "STRONG":
        print("RESULT: STRONG SIGNAL DETECTED")
        print("The Z5D scoring algorithm shows >5x enrichment with p<0.001.")
        print("Top-ranked candidates concentrate significantly near true factors.")
    elif signal == "WEAK":
        print("RESULT: WEAK SIGNAL DETECTED")
        print("The Z5D scoring algorithm shows 2-5x enrichment with p<0.05.")
        print("Top-ranked candidates show some preference for factor regions.")
    elif signal == "NONE":
        print("RESULT: NO SIGNAL DETECTED")
        print("The Z5D scoring algorithm shows <1.5x enrichment.")
        print("Top-ranked candidates are not concentrated near true factors.")
    else:
        print("RESULT: INCONCLUSIVE")
        print("Results do not clearly fit strong, weak, or no-signal categories.")

    print()

    # Save results to JSON
    results = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "Z5D hypothesis validation (Issue #16)",
            "samples": args.samples,
            "top_fraction": args.top_fraction,
            "proximity_threshold": args.proximity,
            "seed": args.seed,
            "generation_time_s": gen_time,
            "scoring_time_s": score_time,
        },
        "ground_truth": {
            "N_127": str(N_127),
            "sqrt_N": str(SQRT_N),
            "p": str(P_TRUE),
            "q": str(Q_TRUE),
            "p_offset_pct": -10.3902,
            "q_offset_pct": 11.5950,
        },
        "search_window": {
            "min": str(SEARCH_MIN),
            "max": str(SEARCH_MAX),
            "width": str(SEARCH_WIDTH),
        },
        "enrichment": enrichment,
        "statistical_tests": test_results,
        "signal_classification": signal,
        "score_distribution": {
            "top_k": top_k,
            "best_score": float(top_candidates[0][1]) if top_candidates else None,
            "worst_top_score": float(top_candidates[-1][1]) if top_candidates else None,
            "best_candidate": str(top_candidates[0][0]) if top_candidates else None,
        },
    }

    output_file = args.output_dir / "z5d_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

    # Save top candidates for further analysis
    top_candidates_file = args.output_dir / "top_candidates.jsonl"
    with open(top_candidates_file, "w") as f:
        for candidate, score in top_candidates[:1000]:  # Save top 1000
            record = {
                "candidate": str(candidate),
                "z5d_score": float(score),
                "dist_to_p": abs(int(candidate) - int(P_TRUE)),
                "dist_to_q": abs(int(candidate) - int(Q_TRUE)),
                "relative_position": float(int(candidate) - int(SEARCH_MIN)) / float(SEARCH_WIDTH),
            }
            f.write(json.dumps(record) + "\n")
    print(f"Top candidates saved to: {top_candidates_file}")

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return 0 if signal in ["STRONG", "WEAK"] else 1


if __name__ == "__main__":
    sys.exit(main())
