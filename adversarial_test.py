#!/usr/bin/env python3
"""
Adversarial Test Suite for Z5D Factorization
Tests on RSA challenge numbers and random semiprimes
"""

import sys
import random
import gmpy2
from gmpy2 import mpz
from sympy import nextprime, isprime
import json
from typing import Tuple, List, Dict
import time

# Import Z5D adapter functions
sys.path.insert(0, '/Users/velocityworks/tmp/copilot/geofac_validation')
from z5d_adapter import z5d_n_est, z5d_predict_nth_prime, compute_z5d_score


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
    
    # Generate two primes of approximately bits//2 each
    half_bits = bits // 2
    
    # Generate random odd number and find next prime
    p_candidate = random.getrandbits(half_bits) | 1
    p = mpz(nextprime(p_candidate))
    
    q_candidate = random.getrandbits(half_bits) | 1
    q = mpz(nextprime(q_candidate))
    
    N = p * q
    
    return N, p, q


def generate_candidates_near_sqrt(N: mpz, k_val: int, num_candidates: int = 100000) -> List[mpz]:
    """
    Generate candidates near sqrt(N) spanning reasonable range.
    Use logarithmic/geometric progression to cover wider range.
    """
    sqrt_N = gmpy2.isqrt(N)
    
    # Determine search range based on bit size
    # For factorization, factors can be up to ~10% away from sqrt(N)
    bits = int(N).bit_length()
    
    # Use percentage-based range (covering ±10% of sqrt(N))
    max_offset_pct = 0.10  # 10%
    max_offset = int(sqrt_N * max_offset_pct)
    
    candidates = []
    
    # Generate candidates in geometric progression for better coverage
    # Split candidates between below and above sqrt(N)
    half = num_candidates // 2
    
    # Below sqrt(N): go from sqrt_N down to sqrt_N - max_offset
    for i in range(half):
        # Geometric progression: denser near sqrt_N, sparser further away
        ratio = float(i) / float(half)
        offset = int(max_offset * ratio)
        candidate = sqrt_N - offset
        
        if candidate > 2:
            candidates.append(candidate)
    
    # Above sqrt(N): go from sqrt_N up to sqrt_N + max_offset
    for i in range(half):
        ratio = float(i) / float(half)
        offset = int(max_offset * ratio)
        candidate = sqrt_N + offset
        
        if candidate < N:
            candidates.append(candidate)
    
    return candidates


def score_candidate(N: mpz, candidate: mpz) -> float:
    """
    Score a candidate factor using Z5D scoring.
    Lower scores are better.
    """
    try:
        # Compute n_est for the candidate
        n_est = z5d_n_est(str(candidate))
        
        # Compute Z5D score
        score = compute_z5d_score(str(candidate), n_est)
        
        return score
    except Exception as e:
        return float('inf')


def test_semiprime(name: str, N: mpz, true_p: mpz, true_q: mpz, 
                   num_candidates: int = 100000) -> Dict:
    """
    Test Z5D on a single semiprime.
    Returns enrichment metrics.
    """
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"N = {N}")
    print(f"True factors: p = {true_p}, q = {true_q}")
    
    sqrt_N = gmpy2.isqrt(N)
    print(f"√N = {sqrt_N}")
    
    # Determine k value (using bit-based heuristic like N₁₂₇)
    bits = int(N).bit_length()
    k_val = max(1, bits // 10)
    
    print(f"Using k = {k_val}, generating {num_candidates} candidates...")
    
    # Generate candidates
    start_time = time.time()
    candidates = generate_candidates_near_sqrt(N, k_val, num_candidates)
    gen_time = time.time() - start_time
    
    print(f"Generated {len(candidates)} candidates in {gen_time:.2f}s")
    
    # Score all candidates
    print("Scoring candidates...")
    start_time = time.time()
    scored = []
    
    for i, c in enumerate(candidates):
        score = score_candidate(N, c)
        scored.append((score, c))
        
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Scored {i+1}/{len(candidates)} ({rate:.1f} candidates/sec)")
    
    score_time = time.time() - start_time
    score_rate = len(candidates) / score_time
    
    print(f"Scored {len(candidates)} candidates in {score_time:.2f}s ({score_rate:.1f} cand/sec)")
    
    # Sort by score (lower is better)
    scored.sort(key=lambda x: x[0])
    
    # Find rank of true factors
    rank_p = None
    rank_q = None
    
    for rank, (score, c) in enumerate(scored, 1):
        if c == true_p:
            rank_p = rank
            print(f"\n✓ Found p at rank {rank_p} with score {score:.6f}")
        if c == true_q:
            rank_q = rank
            print(f"✓ Found q at rank {rank_q} with score {score:.6f}")
    
    if rank_p is None:
        print(f"\n✗ Factor p not found in top {len(candidates)}")
    if rank_q is None:
        print(f"✗ Factor q not found in top {len(candidates)}")
    
    # Calculate enrichment
    # Expected rank if random = len(candidates) / 2
    expected_rank = len(candidates) / 2
    
    # Check top-10K and top-100K
    top_10k = any(c == true_p or c == true_q for _, c in scored[:10000])
    top_100k = any(c == true_p or c == true_q for _, c in scored[:100000])
    
    best_rank = None
    detected_factor = None
    
    if rank_p is not None and rank_q is not None:
        best_rank = min(rank_p, rank_q)
        detected_factor = true_p if rank_p < rank_q else true_q
    elif rank_p is not None:
        best_rank = rank_p
        detected_factor = true_p
    elif rank_q is not None:
        best_rank = rank_q
        detected_factor = true_q
    
    enrichment_10k = None
    enrichment_100k = None
    
    if best_rank is not None:
        # Enrichment = (expected_rank / actual_rank)
        enrichment_10k = min(10000, expected_rank) / min(best_rank, 10000)
        enrichment_100k = min(100000, expected_rank) / min(best_rank, 100000)
    else:
        enrichment_10k = 0.0
        enrichment_100k = 0.0
    
    # Calculate offset from sqrt(N)
    offset_pct = None
    if detected_factor is not None:
        offset = abs(detected_factor - sqrt_N)
        offset_pct = float(offset) / float(sqrt_N) * 100
    
    print(f"\n--- Results ---")
    print(f"Best rank: {best_rank}")
    print(f"Detected factor: {detected_factor}")
    print(f"Offset from √N: {offset_pct:.4f}%" if offset_pct else "Offset: N/A")
    print(f"Top-10K enrichment: {enrichment_10k:.2f}×" if enrichment_10k else "Top-10K: Not found")
    print(f"Top-100K enrichment: {enrichment_100k:.2f}×" if enrichment_100k else "Top-100K: Not found")
    print(f"Scoring rate: {score_rate:.1f} candidates/sec")
    
    return {
        "name": name,
        "N": str(N),
        "true_p": str(true_p),
        "true_q": str(true_q),
        "sqrt_N": str(sqrt_N),
        "detected_factor": str(detected_factor) if detected_factor else None,
        "offset_pct": offset_pct,
        "rank_p": rank_p,
        "rank_q": rank_q,
        "best_rank": best_rank,
        "top_10k": top_10k,
        "top_100k": top_100k,
        "enrichment_10k": enrichment_10k,
        "enrichment_100k": enrichment_100k,
        "score_rate": score_rate,
        "num_candidates": len(candidates)
    }


def main():
    print("="*80)
    print("Z5D ADVERSARIAL TEST SUITE")
    print("="*80)
    
    results = []
    
    # Phase 1: RSA Challenge Numbers
    print("\n\n" + "="*80)
    print("PHASE 1: RSA CHALLENGE NUMBERS")
    print("="*80)
    
    for challenge in RSA_CHALLENGES:
        result = test_semiprime(
            challenge["name"],
            challenge["N"],
            challenge["p"],
            challenge["q"],
            num_candidates=100000
        )
        results.append(result)
    
    # Phase 2: Random Semiprimes
    print("\n\n" + "="*80)
    print("PHASE 2: RANDOM SEMIPRIMES")
    print("="*80)
    
    # Generate 10x 128-bit semiprimes
    for i in range(10):
        N, p, q = gen_semiprime(128, seed=1000 + i)
        result = test_semiprime(
            f"Random-128-{i+1}",
            N, p, q,
            num_candidates=100000
        )
        results.append(result)
    
    # Generate 10x 256-bit semiprimes
    for i in range(10):
        N, p, q = gen_semiprime(256, seed=2000 + i)
        result = test_semiprime(
            f"Random-256-{i+1}",
            N, p, q,
            num_candidates=100000
        )
        results.append(result)
    
    # Summary Report
    print("\n\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    # Separate RSA and Random results
    rsa_results = [r for r in results if r["name"].startswith("RSA")]
    random_results = [r for r in results if r["name"].startswith("Random")]
    
    def print_summary(results, phase_name):
        print(f"\n{phase_name}:")
        print("-" * 80)
        
        enrichments_10k = [r["enrichment_10k"] for r in results if r["enrichment_10k"] is not None and r["enrichment_10k"] > 0]
        enrichments_100k = [r["enrichment_100k"] for r in results if r["enrichment_100k"] is not None and r["enrichment_100k"] > 0]
        
        if enrichments_10k:
            median_10k = sorted(enrichments_10k)[len(enrichments_10k)//2]
            print(f"Median Top-10K enrichment: {median_10k:.2f}×")
        else:
            print("Median Top-10K enrichment: N/A (no factors found)")
        
        if enrichments_100k:
            median_100k = sorted(enrichments_100k)[len(enrichments_100k)//2]
            print(f"Median Top-100K enrichment: {median_100k:.2f}×")
        else:
            print("Median Top-100K enrichment: N/A (no factors found)")
        
        # Count successes
        found_in_10k = sum(1 for r in results if r["top_10k"])
        found_in_100k = sum(1 for r in results if r["top_100k"])
        
        print(f"Found in Top-10K: {found_in_10k}/{len(results)}")
        print(f"Found in Top-100K: {found_in_100k}/{len(results)}")
        
        # Detailed table
        print("\nDetailed Results:")
        print(f"{'Name':<20} {'Enrichment 10K':<18} {'Enrichment 100K':<18} {'Best Rank':<12}")
        print("-" * 80)
        for r in results:
            enr_10k = f"{r['enrichment_10k']:.2f}×" if r['enrichment_10k'] else "Not found"
            enr_100k = f"{r['enrichment_100k']:.2f}×" if r['enrichment_100k'] else "Not found"
            rank = str(r['best_rank']) if r['best_rank'] else "N/A"
            print(f"{r['name']:<20} {enr_10k:<18} {enr_100k:<18} {rank:<12}")
    
    print_summary(rsa_results, "Phase 1: RSA Challenges")
    print_summary(random_results, "Phase 2: Random Semiprimes")
    
    # Save results to JSON
    with open('adversarial_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nFull results saved to: adversarial_results.json")
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    rsa_10k = [r["enrichment_10k"] for r in rsa_results if r["enrichment_10k"] is not None and r["enrichment_10k"] > 0]
    random_10k = [r["enrichment_10k"] for r in random_results if r["enrichment_10k"] is not None and r["enrichment_10k"] > 0]
    
    if rsa_10k:
        rsa_median = sorted(rsa_10k)[len(rsa_10k)//2]
    else:
        rsa_median = 0.0
    
    if random_10k:
        random_median = sorted(random_10k)[len(random_10k)//2]
    else:
        random_median = 0.0
    
    print(f"\nPhase 1 (RSA) median enrichment: {rsa_median:.2f}×")
    print(f"Phase 2 (Random) median enrichment: {random_median:.2f}×")
    
    # Apply success criteria
    rsa_pass = rsa_median >= 3.0
    random_pass = random_median >= 5.0
    rsa_count_pass = sum(1 for e in rsa_10k if e > 2.0) >= 4
    
    rsa_fail = rsa_median < 2.0
    random_fail = random_median < 3.0
    rsa_count_fail = sum(1 for e in rsa_10k if e < 1.5) > 2
    
    print("\nSuccess Criteria:")
    print(f"  RSA median ≥ 3×: {'✓ PASS' if rsa_pass else '✗ FAIL'}")
    print(f"  Random median ≥ 5×: {'✓ PASS' if random_pass else '✗ FAIL'}")
    print(f"  ≥4 RSA challenges >2×: {'✓ PASS' if rsa_count_pass else '✗ FAIL'}")
    
    if rsa_pass and random_pass and rsa_count_pass:
        print("\n🎉 OVERALL: PASS - Real structure detected!")
    elif rsa_fail or random_fail or rsa_count_fail:
        print("\n❌ OVERALL: FAIL - N₁₂₇ was likely lucky/overfit")
    else:
        print("\n⚠️  OVERALL: INCONCLUSIVE - Interesting but not threat-level")


if __name__ == "__main__":
    main()
