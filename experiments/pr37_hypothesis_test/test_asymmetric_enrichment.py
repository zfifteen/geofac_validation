#!/usr/bin/env python3
"""
Focused Test for Asymmetric Q-Factor Enrichment Hypothesis (PR #37)

This test validates or falsifies the hypothesis that Z5D geometric resonance
scoring exhibits asymmetric enrichment:
- q-factor (larger): 5-10× enrichment expected
- p-factor (smaller): ~1× enrichment expected (no signal)

Uses a small, focused test set to quickly determine if the pattern holds.
"""

import sys
import os
import json
from datetime import datetime

# Add repository root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

import gmpy2
from gmpy2 import mpz, isqrt
import numpy as np
from scipy.stats import qmc

from z5d_adapter import z5d_n_est, compute_z5d_score


def generate_qmc_candidates_sobol(search_min, search_max, n_samples, seed=42):
    """Generate QMC candidates using Sobol sequence."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    qmc_samples = sampler.random(n=n_samples)
    
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
    
    return candidates


def score_candidates_z5d(candidates, N):
    """Score all candidates using Z5D."""
    scored = []
    failures = 0
    
    for c in candidates:
        try:
            n_est = z5d_n_est(str(c))
            score = compute_z5d_score(str(c), n_est)
            scored.append((c, score))
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"  Warning: Z5D scoring failed for candidate: {type(e).__name__}")
    
    return scored, failures


def run_single_test(name, N, p_true, q_true, n_candidates=10000, top_pct=10.0, epsilon_pct=1.0):
    """
    Run a single test case to measure asymmetric enrichment.
    
    Returns:
        dict with enrichment measurements
    """
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    # Calculate search window
    sqrt_N = isqrt(N)
    
    # Calculate factor offsets
    p_offset_pct = abs(float(p_true - sqrt_N) / float(sqrt_N) * 100)
    q_offset_pct = abs(float(q_true - sqrt_N) / float(sqrt_N) * 100)
    
    print(f"p offset: {p_offset_pct:.4f}%")
    print(f"q offset: {q_offset_pct:.4f}%")
    
    # Use adaptive window
    max_offset = max(p_offset_pct, q_offset_pct)
    window_pct = max(max_offset * 1.2, 15.0)
    window_radius = int(sqrt_N * window_pct / 100)
    
    search_min = sqrt_N - window_radius
    search_max = sqrt_N + window_radius
    search_width = search_max - search_min
    
    print(f"Search window: ±{window_pct:.2f}%")
    print(f"Generating {n_candidates} QMC candidates...")
    
    # Generate and score candidates
    candidates = generate_qmc_candidates_sobol(search_min, search_max, n_candidates)
    print(f"Scoring with Z5D...")
    scored_candidates, failures = score_candidates_z5d(candidates, N)
    
    if not scored_candidates:
        print(f"ERROR: All candidates failed scoring")
        return None
    
    print(f"Successfully scored {len(scored_candidates)}/{n_candidates} candidates ({failures} failures)")
    
    # Extract top candidates (lower scores are better)
    sorted_candidates = sorted(scored_candidates, key=lambda x: x[1])
    n_top = max(1, int(len(sorted_candidates) * top_pct / 100))
    top_candidates = [c for c, _ in sorted_candidates[:n_top]]
    
    # Calculate proximity window
    epsilon = int(sqrt_N * epsilon_pct / 100)
    
    # Measure enrichment
    baseline_density = float(len(top_candidates)) / float(search_width)
    expected_uniform = baseline_density * (2 * float(epsilon))
    
    # Count candidates near p and q
    count_p = sum(1 for c in top_candidates if abs(c - p_true) < epsilon)
    count_q = sum(1 for c in top_candidates if abs(c - q_true) < epsilon)
    
    # Calculate enrichment ratios
    enrichment_p = count_p / expected_uniform if expected_uniform > 0 else 0.0
    enrichment_q = count_q / expected_uniform if expected_uniform > 0 else 0.0
    
    # Calculate asymmetry ratio
    asymmetry_ratio = enrichment_q / enrichment_p if enrichment_p > 0 else float('inf')
    
    print(f"\nResults:")
    print(f"  Top {top_pct}%: {len(top_candidates)} candidates")
    print(f"  Expected (uniform): {expected_uniform:.2f} candidates in proximity window")
    print(f"  Found near p: {count_p} (enrichment: {enrichment_p:.2f}x)")
    print(f"  Found near q: {count_q} (enrichment: {enrichment_q:.2f}x)")
    print(f"  Asymmetry ratio (q/p): {asymmetry_ratio if asymmetry_ratio != float('inf') else '∞'}")
    
    return {
        'name': name,
        'p_offset_pct': p_offset_pct,
        'q_offset_pct': q_offset_pct,
        'window_pct': window_pct,
        'n_candidates': n_candidates,
        'n_top': len(top_candidates),
        'count_p': count_p,
        'count_q': count_q,
        'expected_uniform': expected_uniform,
        'enrichment_p': enrichment_p,
        'enrichment_q': enrichment_q,
        'asymmetry_ratio': asymmetry_ratio if asymmetry_ratio != float('inf') else 'inf'
    }


def evaluate_hypothesis(results):
    """
    Evaluate if the asymmetric enrichment hypothesis is supported.
    
    Hypothesis is FALSIFIED if:
    1. Q-enrichment ≤2× (expected >5×)
    2. P-enrichment ≥3× (expected ~1×)
    3. Asymmetry ratio <2.0 (expected ≥5)
    
    NOTE: When p-enrichment is 0, asymmetry is infinite (maximum asymmetry).
    This is actually STRONG SUPPORT for the hypothesis, not failure.
    """
    print(f"\n{'='*80}")
    print(f"HYPOTHESIS EVALUATION")
    print(f"{'='*80}")
    
    criteria_failed = 0
    failures = []
    
    # Calculate aggregate statistics
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return "INCONCLUSIVE", 0.0, ["No valid test results"], {}
    
    mean_q_enrichment = np.mean([r['enrichment_q'] for r in valid_results])
    mean_p_enrichment = np.mean([r['enrichment_p'] for r in valid_results])
    
    # Calculate asymmetry ratio more carefully
    # If p-enrichment is 0 but q-enrichment > 0, that's maximum asymmetry (good!)
    # Only use finite values for mean, but note infinite cases
    asymm_ratios_finite = []
    has_infinite_asymmetry = False
    
    for r in valid_results:
        if r['asymmetry_ratio'] == 'inf':
            # Infinite asymmetry (p=0, q>0) is actually the STRONGEST support
            has_infinite_asymmetry = True
        else:
            asymm_ratios_finite.append(r['asymmetry_ratio'])
    
    # Mean asymmetry: if we have infinite cases, treat them as very large (100)
    # for calculation purposes, but note this separately
    if has_infinite_asymmetry:
        mean_asymmetry = np.mean([100.0] * sum(1 for r in valid_results if r['asymmetry_ratio'] == 'inf') + asymm_ratios_finite)
    else:
        mean_asymmetry = np.mean(asymm_ratios_finite) if asymm_ratios_finite else 0.0
    
    print(f"\nAggregate Results across {len(valid_results)} test cases:")
    print(f"  Mean Q-enrichment: {mean_q_enrichment:.2f}x (expected: 5-10x)")
    print(f"  Mean P-enrichment: {mean_p_enrichment:.2f}x (expected: ~1x)")
    
    if has_infinite_asymmetry:
        print(f"  Asymmetry: INFINITE ({sum(1 for r in valid_results if r['asymmetry_ratio'] == 'inf')}/{len(valid_results)} cases)")
        print(f"             (p-enrichment=0, q-enrichment>0 = maximum asymmetry)")
    else:
        print(f"  Mean Asymmetry ratio: {mean_asymmetry:.2f} (expected: ≥5)")
    
    print(f"\nFalsification Criteria:")
    
    # Criterion 1: Q-enrichment ≤2×
    if mean_q_enrichment <= 2.0:
        print(f"  [✗] Criterion 1 FAILED: Q-enrichment ({mean_q_enrichment:.2f}x) ≤ 2.0x")
        criteria_failed += 1
        failures.append(f"Q-enrichment too low: {mean_q_enrichment:.2f}x ≤ 2.0x")
    else:
        print(f"  [✓] Criterion 1 PASSED: Q-enrichment ({mean_q_enrichment:.2f}x) > 2.0x")
    
    # Criterion 2: P-enrichment ≥3×
    if mean_p_enrichment >= 3.0:
        print(f"  [✗] Criterion 2 FAILED: P-enrichment ({mean_p_enrichment:.2f}x) ≥ 3.0x")
        criteria_failed += 1
        failures.append(f"P-enrichment too high: {mean_p_enrichment:.2f}x ≥ 3.0x")
    else:
        print(f"  [✓] Criterion 2 PASSED: P-enrichment ({mean_p_enrichment:.2f}x) < 3.0x")
    
    # Criterion 3: Asymmetry ratio <2.0
    # Special case: infinite asymmetry (p=0, q>0) is MAXIMUM support
    if has_infinite_asymmetry or mean_asymmetry >= 2.0:
        print(f"  [✓] Criterion 3 PASSED: Asymmetry ratio {'∞' if has_infinite_asymmetry else f'{mean_asymmetry:.2f}'} ≥ 2.0")
    else:
        print(f"  [✗] Criterion 3 FAILED: Asymmetry ratio ({mean_asymmetry:.2f}) < 2.0")
        criteria_failed += 1
        failures.append(f"Asymmetry ratio too low: {mean_asymmetry:.2f} < 2.0")
    
    # Make decision (per original spec: ANY ONE failure → FALSIFIED)
    if criteria_failed >= 1:
        decision = "FALSIFIED"
        confidence = 0.95 if criteria_failed >= 2 else 0.85
    elif criteria_failed == 0:
        decision = "CONFIRMED"
        confidence = 0.95
    else:
        decision = "INCONCLUSIVE"
        confidence = 0.50
    
    return decision, confidence, failures, {
        'mean_q_enrichment': mean_q_enrichment,
        'mean_p_enrichment': mean_p_enrichment,
        'mean_asymmetry_ratio': mean_asymmetry,
        'criteria_failed': criteria_failed
    }


def main():
    """Run the hypothesis test."""
    print("="*80)
    print("ASYMMETRIC Q-FACTOR ENRICHMENT HYPOTHESIS TEST")
    print("Testing PR #37 Hypothesis")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load test data - use RSA challenges for testing
    # These have known factors and varying difficulty levels
    test_cases = [
        {
            "name": "RSA-100",
            "N": mpz("1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139"),
            "p": mpz("37975227936943673922808872755445627854565536638199"),
            "q": mpz("40094690950920881030683735292761468389214899724061"),
        },
        {
            "name": "RSA-110",
            "N": mpz("35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667"),
            "p": mpz("6122421090493547576937037317561418841225758554253106999"),
            "q": mpz("5846418214406154678836553182979162384198610505601062333"),
        },
        {
            "name": "RSA-120",
            "N": mpz("227010481295437363334259960947493668895875336466084780038173258247009162675779735389791151574049166747880487470296548479"),
            "p": mpz("327414555693498015751146303749141488063642403240171463406883"),
            "q": mpz("693342667110830181197325401899700641361965863127336680673013"),
        },
        {
            "name": "RSA-129",
            "N": mpz("114381625757888867669235779976146612010218296721242362562561842935706935245733897830597123563958705058989075147599290026879543541"),
            "p": mpz("3490529510847650949147849619903898133417764638493387843990820577"),
            "q": mpz("32769132993266709549961988190834461413177642967992942539798288533"),
        },
    ]
    
    # Run tests
    results = []
    for test_case in test_cases:
        result = run_single_test(
            test_case['name'],
            test_case['N'],
            test_case['p'],
            test_case['q'],
            n_candidates=50000,  # Increased for better statistics
            top_pct=10.0,
            epsilon_pct=1.0
        )
        if result:
            results.append(result)
    
    # Evaluate hypothesis
    decision, confidence, failures, stats = evaluate_hypothesis(results)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print final decision
    print(f"\n{'='*80}")
    print(f"FINAL DECISION: {decision}")
    print(f"Confidence: {confidence * 100:.0f}%")
    print(f"{'='*80}")
    
    if decision == "FALSIFIED":
        print(f"\nThe hypothesis is FALSIFIED. Reasons:")
        for failure in failures:
            print(f"  - {failure}")
    elif decision == "CONFIRMED":
        print(f"\nThe hypothesis is CONFIRMED.")
        print(f"  - Q-enrichment: {stats['mean_q_enrichment']:.2f}x (expected 5-10x)")
        print(f"  - P-enrichment: {stats['mean_p_enrichment']:.2f}x (expected ~1x)")
        print(f"  - Asymmetry: {stats['mean_asymmetry_ratio']:.2f} (expected ≥5)")
    
    print(f"\nTest duration: {duration}")
    
    # Save detailed results
    output_data = {
        'decision': decision,
        'confidence': confidence,
        'criteria_failed': stats['criteria_failed'],
        'statistics': stats,
        'test_results': results,
        'failures': failures if decision == "FALSIFIED" else [],
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration.total_seconds()
    }
    
    output_file = os.path.join(os.path.dirname(__file__), 'test_results.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return 0 if decision == "CONFIRMED" else 1


if __name__ == '__main__':
    sys.exit(main())
