#!/usr/bin/env python3
"""
ADAPTIVE WINDOW Adversarial Test - Fixes PR#25 and PR#27

Key fix: Calculate window from ground truth factors + 20% margin.
This ensures we test Z5D scoring ability, not just window coverage.

Tests the actual hypothesis: "Does Z5D enrich near factors when they're in range?"
"""

import sys
import gmpy2
from gmpy2 import mpz, isqrt
import numpy as np
from scipy.stats import qmc
import time
import json

sys.path.insert(0, '.')
from z5d_adapter import z5d_n_est, compute_z5d_score

# RSA Challenge Numbers
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
]


def calculate_adaptive_window(N, p, q):
    """
    Calculate window that ensures both factors are well within range.
    Uses Option 4: Ground truth + 20% margin
    """
    sqrt_N = isqrt(N)
    
    # Calculate factor offsets as percentage
    p_offset_pct = abs(float(p - sqrt_N) / float(sqrt_N) * 100)
    q_offset_pct = abs(float(q - sqrt_N) / float(sqrt_N) * 100)
    
    # Take max offset and add 20% margin
    max_offset = max(p_offset_pct, q_offset_pct)
    window_pct = max_offset * 1.2  # 20% margin
    
    # Ensure minimum window for statistical significance
    window_pct = max(window_pct, 15.0)  # At least ±15%
    
    window_radius = int(sqrt_N * window_pct / 100)
    
    return window_radius, window_pct


def test_with_adaptive_window(name, N, p_true, q_true, num_candidates=100000):
    """
    Test Z5D with adaptive window that ensures factors are in range.
    Uses PR#20's exact methodology for scoring/enrichment.
    """
    
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    sqrt_N = isqrt(N)
    
    # Calculate offsets
    p_offset = float(p_true - sqrt_N) / float(sqrt_N) * 100
    q_offset = float(q_true - sqrt_N) / float(sqrt_N) * 100
    
    print(f"√N = {sqrt_N}")
    print(f"p offset: {p_offset:.4f}%")
    print(f"q offset: {q_offset:.4f}%")
    
    # Adaptive window calculation
    window_radius, window_pct = calculate_adaptive_window(N, p_true, q_true)
    search_min = sqrt_N - window_radius
    search_max = sqrt_N + window_radius
    search_width = search_max - search_min
    
    print(f"\nAdaptive Window:")
    print(f"  Window: ±{window_pct:.2f}%")
    print(f"  Radius: {window_radius}")
    print(f"  Range: [{search_min}, {search_max}]")
    
    # Verify factors are in window
    p_in = search_min <= p_true <= search_max
    q_in = search_min <= q_true <= search_max
    
    print(f"  p in window: {p_in} ✓" if p_in else f"  p in window: {p_in} ✗")
    print(f"  q in window: {q_in} ✓" if q_in else f"  q in window: {q_in} ✗")
    
    if not (p_in and q_in):
        print("\n⚠️  ERROR: Adaptive window failed to include factors!")
        return None
    
    # Generate candidates via QMC (matching PR#20)
    print(f"\nGenerating {num_candidates:,} candidates via QMC...")
    start = time.time()
    
    sampler = qmc.Sobol(d=2, scramble=True, seed=42)
    qmc_samples = sampler.random(n=num_candidates)
    
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
    
    gen_time = time.time() - start
    print(f"Generated in {gen_time:.2f}s")
    
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
            print(f"  {i+1:,}/{num_candidates:,} ({rate:.0f} cand/sec)")
    
    score_time = time.time() - start
    score_rate = num_candidates / score_time
    print(f"Scored {num_candidates:,} in {score_time:.2f}s ({score_rate:.0f} cand/sec)")
    
    # Sort by score (lower is better)
    scored.sort(key=lambda x: x[1])
    
    # Take top 10% (matching PR#20)
    top_k = num_candidates // 10
    top_candidates = [c for c, _ in scored[:top_k]]
    
    # Calculate enrichment SEPARATELY for p and q (critical fix from post-mortem!)
    def calc_enrichment(cands, target, search_width_val, threshold_pct=0.01):
        """Calculate % of candidates within threshold of target"""
        threshold = float(search_width_val) * threshold_pct
        near = sum(1 for c in cands if abs(float(c) - float(target)) < threshold)
        return (near / len(cands)) * 100
    
    # Baseline (all candidates)
    baseline_near_p = calc_enrichment(candidates, p_true, search_width)
    baseline_near_q = calc_enrichment(candidates, q_true, search_width)
    
    # Top-K
    top_near_p = calc_enrichment(top_candidates, p_true, search_width)
    top_near_q = calc_enrichment(top_candidates, q_true, search_width)
    
    # Enrichment factors
    enr_p = top_near_p / baseline_near_p if baseline_near_p > 0 else 0
    enr_q = top_near_q / baseline_near_q if baseline_near_q > 0 else 0
    
    print(f"\n{'='*80}")
    print("RESULTS (±1% threshold)")
    print(f"{'='*80}")
    print()
    
    print(f"Baseline (all {num_candidates:,}):")
    print(f"  Near p: {baseline_near_p:.4f}%")
    print(f"  Near q: {baseline_near_q:.4f}%")
    print()
    
    print(f"Top {top_k:,} Z5D-scored:")
    print(f"  Near p: {top_near_p:.4f}%")
    print(f"  Near q: {top_near_q:.4f}%")
    print()
    
    print(f"Enrichment:")
    print(f"  p: {enr_p:.2f}×")
    print(f"  q: {enr_q:.2f}×")
    print()
    
    # Classify result (matching N₁₂₇ criteria)
    max_enr = max(enr_p, enr_q)
    
    if max_enr >= 5.0:
        if enr_p >= 5.0 and enr_q >= 5.0:
            result = "✓ STRONG - Both factors enriched"
        else:
            result = "✓ STRONG - Asymmetric (like N₁₂₇)"
    elif max_enr >= 2.0:
        result = "⚠️  WEAK - Some enrichment"
    else:
        result = "✗ NONE - No enrichment"
    
    print(f"Result: {result}")
    
    return {
        "name": name,
        "window_pct": window_pct,
        "p_offset_pct": p_offset,
        "q_offset_pct": q_offset,
        "num_candidates": num_candidates,
        "top_k": top_k,
        "baseline_near_p": baseline_near_p,
        "baseline_near_q": baseline_near_q,
        "top_near_p": top_near_p,
        "top_near_q": top_near_q,
        "enrichment_p": enr_p,
        "enrichment_q": enr_q,
        "result": result,
        "score_rate": score_rate
    }


def main():
    print("="*80)
    print("ADAPTIVE WINDOW ADVERSARIAL TEST")
    print("Fixes PR#25 and PR#27 Window Coverage Issue")
    print("="*80)
    print()
    print("Key Innovation:")
    print("  - Adaptive window calculated from ground truth + 20% margin")
    print("  - Ensures factors are always in range")
    print("  - Tests: 'Does Z5D enrich near factors when they're reachable?'")
    print()
    print("This addresses the core issue identified in post-mortem:")
    print("  Fixed ±13% worked for N₁₂₇ (factors at ±10-11%)")
    print("  Fixed ±13% failed for RSA (factors at ±30-200%)")
    print("  Solution: Adapt window to factor positions")
    print()
    
    results = []
    
    for test in RSA_CHALLENGES:
        result = test_with_adaptive_window(
            test["name"],
            test["N"],
            test["p"],
            test["q"],
            num_candidates=100000
        )
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print()
    
    print(f"{'Test':<12} {'Window':<10} {'p enrich':<12} {'q enrich':<12} {'Result'}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<12} ±{r['window_pct']:>5.1f}% {r['enrichment_p']:>8.2f}× {r['enrichment_q']:>8.2f}× {r['result']}")
    
    print()
    print("Comparison to N₁₂₇ (±13% window):")
    print("  N₁₂₇: p=0.00×, q=10.00× (asymmetric strong signal)")
    print()
    
    # Count successes
    strong = sum(1 for r in results if 'STRONG' in r['result'])
    weak = sum(1 for r in results if 'WEAK' in r['result'])
    none = sum(1 for r in results if 'NONE' in r['result'])
    
    print(f"Results:")
    print(f"  Strong signal: {strong}/{len(results)}")
    print(f"  Weak signal: {weak}/{len(results)}")
    print(f"  No signal: {none}/{len(results)}")
    print()
    
    if strong > 0:
        print("✓ Z5D shows enrichment when factors are in range!")
        print("  Previous 'falsification' was due to fixed window limitation.")
    elif weak > 0:
        print("⚠️  Weak signal detected - Z5D may work but needs optimization")
    else:
        print("✗ No enrichment even with adaptive windows")
        print("  Z5D may not generalize beyond N₁₂₇")
    
    # Save results
    with open('adaptive_window_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: adaptive_window_results.json")


if __name__ == "__main__":
    main()
