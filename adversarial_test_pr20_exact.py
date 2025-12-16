#!/usr/bin/env python3
"""
EXACT REPLICATION of PR#20 methodology on RSA challenges

This matches PR#20's N₁₂₇ test EXACTLY:
- 100,000 candidates (not 10K)
- ±13% window
- QMC/Sobol generation
- Separate p and q enrichment analysis
- Top-10K (10%) vs baseline comparison
"""

import sys
import gmpy2
from gmpy2 import mpz, isqrt
import numpy as np
from scipy.stats import qmc
import time

sys.path.insert(0, '.')
from z5d_adapter import z5d_n_est, compute_z5d_score

# Test just RSA-100 and RSA-110 (the ones within ±13%)
RSA_TESTS = [
    {
        "name": "RSA-100",
        "N": mpz("1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139"),
        "p": mpz("37975227936943673922808872755445627854565536638199"),
        "q": mpz("40094690950920881030683735292761468389214899724061")
    },
    {
        "name": "RSA-110",
        "N": mpz("35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667"),
        "p": mpz("6122421090493547576937037317561418841225758554253106999"),
        "q": mpz("5846418214406154678836553182979162384198610505601062333")
    }
]


def test_semiprime_pr20_exact(name, N, p_true, q_true):
    """Exact replication of PR#20 methodology"""
    
    print(f"\n{'='*80}")
    print(f"Testing: {name} (PR#20 Exact Methodology)")
    print(f"{'='*80}")
    
    sqrt_N = isqrt(N)
    
    print(f"√N = {sqrt_N}")
    print(f"p = {p_true}")
    print(f"q = {q_true}")
    
    # Calculate offsets
    p_offset = float(p_true - sqrt_N) / float(sqrt_N) * 100
    q_offset = float(q_true - sqrt_N) / float(sqrt_N) * 100
    
    print(f"p offset: {p_offset:.4f}%")
    print(f"q offset: {q_offset:.4f}%")
    
    # Identify which is larger
    if abs(q_offset) > abs(p_offset):
        print(f"q is FARTHER from √N (|{q_offset:.2f}%| > |{p_offset:.2f}%|)")
        larger_factor = q_true
        smaller_factor = p_true
    else:
        print(f"p is FARTHER from √N (|{p_offset:.2f}%| > |{q_offset:.2f}%|)")
        larger_factor = p_true
        smaller_factor = q_true
    
    # ±13% window (EXACT match to PR#20)
    window_radius = (sqrt_N * 13) // 100
    search_min = sqrt_N - window_radius
    search_max = sqrt_N + window_radius
    search_width = search_max - search_min
    
    # Check if factors in window
    p_in = search_min <= p_true <= search_max
    q_in = search_min <= q_true <= search_max
    
    print(f"p in ±13% window: {p_in}")
    print(f"q in ±13% window: {q_in}")
    
    if not (p_in and q_in):
        print("\n⚠️  Factors outside window - SKIPPING")
        return None
    
    # Generate 100,000 candidates (EXACT match to PR#20)
    print(f"\nGenerating 100,000 candidates via QMC...")
    start = time.time()
    
    sampler = qmc.Sobol(d=2, scramble=True, seed=42)
    qmc_samples = sampler.random(n=100000)
    
    search_min_int = int(search_min)
    search_max_int = int(search_max)
    search_range_int = search_max_int - search_min_int
    
    scale = 1 << 53
    denom_bits = 106
    
    candidates = []
    for row in qmc_samples:
        hi = min(int(row[0] * scale), scale - 1)
        lo = min(int(row[1] * scale), scale - 1)
        x = (hi << 53) | lo
        offset = (x * (search_range_int + 1)) >> denom_bits
        candidate = search_min_int + offset
        if candidate % 2 == 0:
            candidate += 1
            if candidate > search_max_int:
                candidate -= 2
        candidates.append(mpz(candidate))
    
    print(f"Generated {len(candidates):,} in {time.time()-start:.2f}s")
    
    # Score all candidates
    print("Scoring...")
    start = time.time()
    scored = []
    
    for i, c in enumerate(candidates):
        try:
            n_est = z5d_n_est(str(c))
            score = compute_z5d_score(str(c), n_est)
            scored.append((c, score))
        except:
            scored.append((c, float('inf')))
        
        if (i + 1) % 10000 == 0:
            rate = (i + 1) / (time.time() - start)
            print(f"  {i+1:,}/100,000 ({rate:.0f} cand/sec)")
    
    print(f"Scored 100,000 in {time.time()-start:.2f}s")
    
    # Sort by score
    scored.sort(key=lambda x: x[1])
    
    # Take top 10,000 (10% - EXACT match to PR#20)
    top_10k = [c for c, _ in scored[:10000]]
    
    # Calculate enrichment SEPARATELY for p and q (like PR#20)
    def calc_enrichment(candidates, target, search_width_val, threshold_pct=0.01):
        """Calculate % of candidates within threshold of target"""
        threshold = float(search_width_val) * threshold_pct
        near = sum(1 for c in candidates if abs(float(c) - float(target)) < threshold)
        return (near / len(candidates)) * 100
    
    # Baseline: all 100K candidates
    baseline_near_p = calc_enrichment(candidates, p_true, search_width)
    baseline_near_q = calc_enrichment(candidates, q_true, search_width)
    
    # Top 10K
    top_near_p = calc_enrichment(top_10k, p_true, search_width)
    top_near_q = calc_enrichment(top_10k, q_true, search_width)
    
    # Enrichment factors
    enr_p = top_near_p / baseline_near_p if baseline_near_p > 0 else 0
    enr_q = top_near_q / baseline_near_q if baseline_near_q > 0 else 0
    
    print(f"\n{'='*80}")
    print("RESULTS (±1% threshold)")
    print(f"{'='*80}")
    print()
    
    print(f"Baseline (all 100K):")
    print(f"  Near p: {baseline_near_p:.4f}%")
    print(f"  Near q: {baseline_near_q:.4f}%")
    print()
    
    print(f"Top 10K Z5D-scored:")
    print(f"  Near p: {top_near_p:.4f}%")
    print(f"  Near q: {top_near_q:.4f}%")
    print()
    
    print(f"Enrichment:")
    print(f"  p: {enr_p:.2f}×")
    print(f"  q: {enr_q:.2f}×")
    print()
    
    # Classify result
    if enr_q >= 5.0 and enr_p < 2.0:
        result = "✓ ASYMMETRIC SUCCESS (q enriched like N₁₂₇)"
    elif enr_q >= 5.0 and enr_p >= 5.0:
        result = "✓ SYMMETRIC SUCCESS (both enriched)"
    elif enr_q >= 2.0 or enr_p >= 2.0:
        result = "⚠️  WEAK SIGNAL"
    else:
        result = "✗ NO SIGNAL"
    
    print(f"Result: {result}")
    
    return {
        "name": name,
        "p_offset_pct": p_offset,
        "q_offset_pct": q_offset,
        "baseline_near_p": baseline_near_p,
        "baseline_near_q": baseline_near_q,
        "top_near_p": top_near_p,
        "top_near_q": top_near_q,
        "enrichment_p": enr_p,
        "enrichment_q": enr_q,
        "result": result
    }


def main():
    print("="*80)
    print("EXACT PR#20 REPLICATION ON RSA CHALLENGES")
    print("="*80)
    print()
    print("Configuration (matches PR#20 exactly):")
    print("  - 100,000 candidates via QMC/Sobol")
    print("  - ±13% search window")
    print("  - Top 10,000 (10%) analyzed")
    print("  - Separate p and q enrichment at ±1%")
    print()
    
    results = []
    for test in RSA_TESTS:
        result = test_semiprime_pr20_exact(
            test["name"],
            test["N"],
            test["p"],
            test["q"]
        )
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print()
    
    print(f"{'Test':<15} {'p enrich':<12} {'q enrich':<12} {'Result'}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<15} {r['enrichment_p']:>8.2f}× {r['enrichment_q']:>8.2f}× {r['result']}")
    
    print()
    print("Comparison to N₁₂₇:")
    print("  N₁₂₇: p=0.00×, q=10.00× (asymmetric success)")
    print()
    
    if any(r['enrichment_q'] >= 5.0 for r in results):
        print("✓ At least one RSA challenge shows q-enrichment like N₁₂₇!")
    else:
        print("✗ No RSA challenges show strong q-enrichment")


if __name__ == "__main__":
    main()
