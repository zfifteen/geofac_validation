#!/usr/bin/env python3
"""
CORRECTED Adversarial Test Suite for Z5D Factorization

This uses PR#20's validated methodology:
1. Search window: ±13% of sqrt(N) (not ±10%)
2. Uniform candidate generation (QMC/Sobol)
3. Proximity enrichment test (distance-based, not rank-based)
4. Statistical comparison: top-scored vs random baseline

Tests on RSA challenge numbers and random semiprimes.
"""

import sys
import random
import gmpy2
from gmpy2 import mpz
from sympy import nextprime
import json
from typing import Tuple, List, Dict
import time
import numpy as np
from scipy.stats import qmc

# Import Z5D adapter functions
sys.path.insert(0, '/Users/velocityworks/tmp/copilot/geofac_validation')
from z5d_adapter import z5d_n_est, compute_z5d_score


# RSA Challenge Numbers with Known Factors
RSA_CHALLENGES = [
    {
        "name": "RSA-100",
        "bits": 330,
        "N": mpz("1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139"),
        "p": mpz("37975227936943673922808872755445627854565536638199"),
        "q": mpz("40094690950920881030683735292761468389214899724061")
    },
    {
        "name": "RSA-110",
        "bits": 364,
        "N": mpz("35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667"),
        "p": mpz("6122421090493547576937037317561418841225758554253106999"),
        "q": mpz("5846418214406154678836553182979162384198610505601062333")
    },
    {
        "name": "RSA-120",
        "bits": 397,
        "N": mpz("227010481295437363334259960947493668895875336466084780038173258247009162675779735389791151574049166747880487470296548479"),
        "p": mpz("327414555693498015751146303749141488063642403240171463406883"),
        "q": mpz("693342667110830181197325401899700641361965863127336680673013")
    },
    {
        "name": "RSA-129",
        "bits": 426,
        "N": mpz("114381625757888867669235779976146612010218296721242362562561842935706935245733897830597123563958705058989075147599290026879543541"),
        "p": mpz("3490529510847650949147849619903898133417764638493387843990820577"),
        "q": mpz("32769132993266709549961988190834461413177642967992942539798288533")
    },
    {
        "name": "RSA-130",
        "bits": 430,
        "N": mpz("1807082088687404805951656164405905566278102516769401349170127021450056662540244048387341127590812303371781887966563182013214880557"),
        "p": mpz("39685999459597454290161126162883786067576449112810064832555157243"),
        "q": mpz("45534498646735972188403686897274408864356301263205069600999044599")
    },
    {
        "name": "RSA-140",
        "bits": 463,
        "N": mpz("21290246318258757547497882016271517497806703963277216278233383215381949984056495911366573853033449030401269848860909346025397551999"),
        "p": mpz("33987174230284385545301237457012937786026211282789250180279807952743889547129"),
        "q": mpz("62640132235440353467565674076428197634962849887148381806811518066862371063")
    }
]


def gen_semiprime(bits: int, seed: int = None) -> Tuple[mpz, mpz, mpz]:
    """Generate a random semiprime of specified bit size"""
    if seed is not None:
        random.seed(seed)
    
    half_bits = bits // 2
    
    p_candidate = random.getrandbits(half_bits) | 1
    p = mpz(nextprime(p_candidate))
    
    q_candidate = random.getrandbits(half_bits) | 1
    q = mpz(nextprime(q_candidate))
    
    N = p * q
    
    return N, p, q


def generate_candidates_qmc(N: mpz, num_candidates: int, seed: int = 42) -> List[mpz]:
    """
    Generate candidates using QMC (Sobol) for uniform coverage.
    Uses ±13% search window matching PR#20.
    
    This is the CORRECT approach from validated N₁₂₇ implementation.
    """
    sqrt_N = gmpy2.isqrt(N)
    
    # CRITICAL: Use ±13% window, not ±10%
    search_radius = (sqrt_N * 13) // 100
    search_min = sqrt_N - search_radius
    search_max = sqrt_N + search_radius
    search_range = search_max - search_min
    
    # Generate QMC samples using Sobol sequence
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    qmc_samples = sampler.random(n=num_candidates)
    
    # Map to search range with high precision
    search_min_int = int(search_min)
    search_max_int = int(search_max)
    search_range_int = search_max_int - search_min_int
    
    scale = 1 << 53
    denom_bits = 106
    
    candidates = []
    for row in qmc_samples:
        # 106-bit precision mapping
        hi = min(int(row[0] * scale), scale - 1)
        lo = min(int(row[1] * scale), scale - 1)
        
        x = (hi << 53) | lo
        offset = (x * (search_range_int + 1)) >> denom_bits
        
        candidate = search_min_int + offset
        
        # Make odd (primes are odd)
        if candidate % 2 == 0:
            candidate += 1
            if candidate > search_max_int:
                candidate -= 2
        
        candidates.append(mpz(candidate))
    
    return candidates


def score_candidate(candidate: mpz) -> float:
    """Score a candidate using Z5D."""
    try:
        n_est = z5d_n_est(str(candidate))
        score = compute_z5d_score(str(candidate), n_est)
        return score
    except Exception:
        return float('inf')


def compute_distance_to_factors(candidates: List[mpz], p_true: mpz, q_true: mpz) -> np.ndarray:
    """
    Compute minimum distance from each candidate to nearest true factor.
    This is the KEY METRIC from PR#20.
    Returns as float array to avoid int overflow in numpy.
    """
    distances = []
    for c in candidates:
        dist_to_p = abs(int(c) - int(p_true))
        dist_to_q = abs(int(c) - int(q_true))
        min_dist = min(dist_to_p, dist_to_q)
        distances.append(float(min_dist))  # Convert to float for numpy
    
    return np.array(distances, dtype=np.float64)


def compute_proximity_enrichment(top_distances: np.ndarray, 
                                 baseline_distances: np.ndarray,
                                 proximity_threshold_pct: float = 0.01) -> Dict:
    """
    Compute proximity enrichment: Are top candidates closer to factors?
    
    Uses PR#20's methodology:
    - Count how many candidates are within proximity_threshold of factors
    - Compare top-scored vs random baseline
    - Calculate enrichment factor
    """
    # Determine absolute threshold
    # We need search width for normalization
    threshold_abs = np.percentile(baseline_distances, 1)  # Use 1st percentile as reference
    
    # Count candidates within threshold
    top_near = np.sum(top_distances <= threshold_abs)
    baseline_near = np.sum(baseline_distances <= threshold_abs)
    
    # Expected count if random
    expected = len(baseline_distances) * (baseline_near / len(baseline_distances))
    
    # Enrichment factor
    if expected > 0:
        enrichment = top_near / expected
    else:
        enrichment = 0.0
    
    # Mean distances
    top_mean = float(np.mean(top_distances))
    baseline_mean = float(np.mean(baseline_distances))
    
    # Distance ratio (lower is better)
    distance_ratio = top_mean / baseline_mean if baseline_mean > 0 else 1.0
    
    return {
        "top_near_count": int(top_near),
        "baseline_near_count": int(baseline_near),
        "expected_count": float(expected),
        "enrichment_factor": float(enrichment),
        "top_mean_distance": top_mean,
        "baseline_mean_distance": baseline_mean,
        "distance_ratio": distance_ratio,
        "threshold_abs": float(threshold_abs)
    }


def test_semiprime(name: str, N: mpz, true_p: mpz, true_q: mpz,
                   num_candidates: int = 100000, top_k: int = 10000) -> Dict:
    """
    Test Z5D on a single semiprime using CORRECTED PR#20 methodology.
    """
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"N = {N}")
    print(f"True factors: p = {true_p}, q = {true_q}")
    
    sqrt_N = gmpy2.isqrt(N)
    print(f"√N = {sqrt_N}")
    
    # Check factor positions
    p_offset_pct = float(true_p - sqrt_N) / float(sqrt_N) * 100
    q_offset_pct = float(true_q - sqrt_N) / float(sqrt_N) * 100
    print(f"p offset: {p_offset_pct:.4f}%")
    print(f"q offset: {q_offset_pct:.4f}%")
    
    # Check if factors are in ±13% window
    search_radius = (sqrt_N * 13) // 100
    search_min = sqrt_N - search_radius
    search_max = sqrt_N + search_radius
    
    p_in_range = search_min <= true_p <= search_max
    q_in_range = search_min <= true_q <= search_max
    
    print(f"p in ±13% window: {p_in_range}")
    print(f"q in ±13% window: {q_in_range}")
    
    if not (p_in_range and q_in_range):
        print(f"\n⚠️  WARNING: Factors outside ±13% window - test may be invalid")
    
    print(f"\nGenerating {num_candidates:,} candidates via QMC...")
    start_time = time.time()
    candidates = generate_candidates_qmc(N, num_candidates)
    gen_time = time.time() - start_time
    print(f"Generated in {gen_time:.2f}s")
    
    # Score all candidates
    print("Scoring candidates...")
    start_time = time.time()
    scored = []
    
    for i, c in enumerate(candidates):
        score = score_candidate(c)
        scored.append((c, score))
        
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Scored {i+1:,}/{num_candidates:,} ({rate:.1f} cand/sec)")
    
    score_time = time.time() - start_time
    score_rate = len(candidates) / score_time
    
    print(f"Scored {len(candidates):,} in {score_time:.2f}s ({score_rate:.1f} cand/sec)")
    
    # Sort by score (lower is better)
    scored.sort(key=lambda x: x[1])
    
    # Take top-k
    top_candidates = [c for c, _ in scored[:top_k]]
    
    # Random baseline
    print(f"\nComparing top {top_k:,} vs random {top_k:,}...")
    np.random.seed(42)
    baseline_indices = np.random.choice(len(candidates), size=top_k, replace=False)
    baseline_candidates = [candidates[i] for i in baseline_indices]
    
    # Compute distances to factors (KEY METRIC)
    top_distances = compute_distance_to_factors(top_candidates, true_p, true_q)
    baseline_distances = compute_distance_to_factors(baseline_candidates, true_p, true_q)
    
    # Proximity enrichment
    enrichment = compute_proximity_enrichment(top_distances, baseline_distances)
    
    print(f"\n--- Results ---")
    print(f"Top mean distance: {enrichment['top_mean_distance']:.4e}")
    print(f"Baseline mean distance: {enrichment['baseline_mean_distance']:.4e}")
    print(f"Distance ratio (top/baseline): {enrichment['distance_ratio']:.4f}×")
    print(f"Enrichment factor: {enrichment['enrichment_factor']:.2f}×")
    
    # Statistical test
    from scipy import stats
    stat, p_value = stats.mannwhitneyu(top_distances, baseline_distances, alternative='less')
    
    print(f"Mann-Whitney U test p-value: {p_value:.4e}")
    print(f"Significant (p<0.001): {p_value < 0.001}")
    
    return {
        "name": name,
        "N": str(N),
        "true_p": str(true_p),
        "true_q": str(true_q),
        "sqrt_N": str(sqrt_N),
        "p_offset_pct": p_offset_pct,
        "q_offset_pct": q_offset_pct,
        "p_in_window": p_in_range,
        "q_in_window": q_in_range,
        "num_candidates": num_candidates,
        "top_k": top_k,
        "score_rate": score_rate,
        **enrichment,
        "mann_whitney_p": float(p_value),
        "significant_001": bool(p_value < 0.001),
        "significant_005": bool(p_value < 0.05)
    }


def main():
    print("="*80)
    print("CORRECTED Z5D ADVERSARIAL TEST SUITE")
    print("Using PR#20 Validated Methodology")
    print("="*80)
    print()
    print("Configuration:")
    print("  - Search window: ±13% of sqrt(N)")
    print("  - Candidate generation: QMC (Sobol)")
    print("  - Test metric: Proximity enrichment (distance-based)")
    print("  - Comparison: Top-10K vs Random-10K")
    print()
    
    results = []
    
    # Phase 1: RSA Challenge Numbers
    print("\n" + "="*80)
    print("PHASE 1: RSA CHALLENGE NUMBERS")
    print("="*80)
    
    for challenge in RSA_CHALLENGES:
        result = test_semiprime(
            challenge["name"],
            challenge["N"],
            challenge["p"],
            challenge["q"],
            num_candidates=100000,
            top_k=10000
        )
        results.append(result)
    
    # Phase 2: Random Semiprimes (smaller set for speed)
    print("\n" + "="*80)
    print("PHASE 2: RANDOM SEMIPRIMES")
    print("="*80)
    
    # 5x 128-bit
    for i in range(5):
        N, p, q = gen_semiprime(128, seed=1000 + i)
        result = test_semiprime(
            f"Random-128-{i+1}",
            N, p, q,
            num_candidates=100000,
            top_k=10000
        )
        results.append(result)
    
    # 5x 256-bit
    for i in range(5):
        N, p, q = gen_semiprime(256, seed=2000 + i)
        result = test_semiprime(
            f"Random-256-{i+1}",
            N, p, q,
            num_candidates=100000,
            top_k=10000
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    rsa_results = [r for r in results if r["name"].startswith("RSA")]
    random_results = [r for r in results if r["name"].startswith("Random")]
    
    def print_summary(results_subset, phase_name):
        print(f"\n{phase_name}:")
        print("-" * 80)
        
        enrichments = [r["enrichment_factor"] for r in results_subset]
        distance_ratios = [r["distance_ratio"] for r in results_subset]
        p_values = [r["mann_whitney_p"] for r in results_subset]
        
        if enrichments:
            median_enr = sorted(enrichments)[len(enrichments)//2]
            median_ratio = sorted(distance_ratios)[len(distance_ratios)//2]
            significant = sum(1 for p in p_values if p < 0.001)
            
            print(f"Median enrichment factor: {median_enr:.2f}×")
            print(f"Median distance ratio: {median_ratio:.4f}×")
            print(f"Significant results (p<0.001): {significant}/{len(results_subset)}")
        
        print(f"\n{'Name':<20} {'Enrichment':<15} {'Dist Ratio':<15} {'P-value':<15}")
        print("-" * 80)
        for r in results_subset:
            print(f"{r['name']:<20} {r['enrichment_factor']:>8.2f}× {r['distance_ratio']:>12.4f}× {r['mann_whitney_p']:>12.4e}")
    
    print_summary(rsa_results, "Phase 1: RSA Challenges")
    print_summary(random_results, "Phase 2: Random Semiprimes")
    
    # Save results
    with open('adversarial_corrected_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: adversarial_corrected_results.json")
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    rsa_enrichments = [r["enrichment_factor"] for r in rsa_results]
    random_enrichments = [r["enrichment_factor"] for r in random_results]
    
    if rsa_enrichments:
        rsa_median = sorted(rsa_enrichments)[len(rsa_enrichments)//2]
    else:
        rsa_median = 0.0
    
    if random_enrichments:
        random_median = sorted(random_enrichments)[len(random_enrichments)//2]
    else:
        random_median = 0.0
    
    print(f"\nPhase 1 (RSA) median enrichment: {rsa_median:.2f}×")
    print(f"Phase 2 (Random) median enrichment: {random_median:.2f}×")
    
    # Apply Issue #24 success criteria
    rsa_pass = rsa_median >= 3.0
    random_pass = random_median >= 5.0
    
    print("\nSuccess Criteria (from Issue #24):")
    print(f"  RSA median ≥3×: {'✓ PASS' if rsa_pass else '✗ FAIL'}")
    print(f"  Random median ≥5×: {'✓ PASS' if random_pass else '✗ FAIL'}")
    
    if rsa_pass and random_pass:
        print("\n✓ OVERALL: PASS - Z5D shows real structure")
    elif rsa_median >= 2.0 or random_median >= 3.0:
        print("\n⚠️  OVERALL: WEAK SIGNAL - Some structure but below threshold")
    else:
        print("\n✗ OVERALL: FAIL - No significant structure")


if __name__ == "__main__":
    main()
