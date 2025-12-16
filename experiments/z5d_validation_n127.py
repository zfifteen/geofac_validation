#!/usr/bin/env python3
"""
Z5D Validation Experiment for N_127

This script tests the falsifiable hypothesis: Does Z5D resonance scoring 
concentrate candidates near the true factors of N_127 better than random sampling?

Ground Truth from PR #6:
    N_127 = 137524771864208156028430259349934309717
    p = 10508623501177419659  (smaller factor, -10.39% below sqrt(N))
    q = 13086849276577416863  (larger factor, +11.59% above sqrt(N))
    sqrt(N_127) ≈ 11727095627827384440

Experiment Design:
1. Generate 100K-1M candidates uniformly in ±13% window around sqrt(N_127)
2. Score each candidate with Z5D resonance
3. Analyze spatial distribution and enrichment near true factors
4. Compare Top-K slices vs random baseline
"""

import sys
import csv
import time
import random
import json
from pathlib import Path
import gmpy2

# Import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from z5d_adapter import z5d_n_est, compute_z5d_score

# Ground truth constants
N_127 = gmpy2.mpz("137524771864208156028430259349934309717")
TRUE_P = gmpy2.mpz("10508623501177419659")
TRUE_Q = gmpy2.mpz("13086849276577416863")
SQRT_N = gmpy2.mpz("11727095627827384440")

# Search window: ±13% around sqrt(N)
WINDOW_RADIUS = (SQRT_N * 13) // 100
SEARCH_MIN = SQRT_N - WINDOW_RADIUS
SEARCH_MAX = SQRT_N + WINDOW_RADIUS

# Experimental parameters - start with 100K for quick validation
NUM_CANDIDATES = 100_000
RANDOM_SEED = 127

# Output files
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_CSV = OUTPUT_DIR / "z5d_validation_n127_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "z5d_validation_n127_summary.json"


def generate_uniform_candidates(n, seed=RANDOM_SEED):
    """
    Generate n candidates uniformly distributed in [SEARCH_MIN, SEARCH_MAX].
    Only odd numbers (since N_127 is odd, factors must be odd).
    
    Returns list of gmpy2.mpz candidates.
    """
    random.seed(seed)
    candidates = []
    
    # Search space size
    space_size = SEARCH_MAX - SEARCH_MIN
    
    print(f"Generating {n:,} uniform candidates in range:")
    print(f"  [{int(SEARCH_MIN)}, {int(SEARCH_MAX)}]")
    window_pct = int(WINDOW_RADIUS * 100 // SQRT_N)
    print(f"  Window: ±{int(WINDOW_RADIUS):,} ({window_pct}% of sqrt(N))")
    
    for i in range(n):
        # Generate random offset, ensure odd
        offset = random.randrange(0, space_size)
        candidate = SEARCH_MIN + offset
        
        # Make odd
        if candidate % 2 == 0:
            candidate += 1
        
        # Ensure still in range
        if candidate > SEARCH_MAX:
            candidate = SEARCH_MAX if SEARCH_MAX % 2 == 1 else SEARCH_MAX - 1
            
        candidates.append(gmpy2.mpz(candidate))
        
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1:,} candidates...")
    
    print(f"Generated {len(candidates):,} candidates")
    return candidates


def compute_distance_metrics(candidate):
    """
    Compute distance metrics for a candidate relative to true factors.
    
    Returns dict with:
        - distance_to_p: absolute distance to smaller factor
        - distance_to_q: absolute distance to larger factor
        - distance_to_nearest: minimum distance to either factor
        - pct_from_sqrt: percentage offset from sqrt(N)
        - is_near_p_1pct: within ±1% of p
        - is_near_q_1pct: within ±1% of q
        - is_near_p_5pct: within ±5% of p
        - is_near_q_5pct: within ±5% of q
    """
    dist_p = abs(candidate - TRUE_P)
    dist_q = abs(candidate - TRUE_Q)
    dist_nearest = min(dist_p, dist_q)
    
    # Percentage from sqrt(N)
    pct_from_sqrt = float((candidate - SQRT_N) * 100) / float(SQRT_N)
    
    # Check proximity thresholds
    threshold_1pct_p = TRUE_P // 100
    threshold_1pct_q = TRUE_Q // 100
    threshold_5pct_p = (TRUE_P * 5) // 100
    threshold_5pct_q = (TRUE_Q * 5) // 100
    
    return {
        'distance_to_p': int(dist_p),
        'distance_to_q': int(dist_q),
        'distance_to_nearest': int(dist_nearest),
        'pct_from_sqrt': pct_from_sqrt,
        'is_near_p_1pct': int(dist_p <= threshold_1pct_p),
        'is_near_q_1pct': int(dist_q <= threshold_1pct_q),
        'is_near_p_5pct': int(dist_p <= threshold_5pct_p),
        'is_near_q_5pct': int(dist_q <= threshold_5pct_q),
        'is_near_any_5pct': int(dist_p <= threshold_5pct_p or dist_q <= threshold_5pct_q)
    }


def score_candidates(candidates):
    """
    Score all candidates with Z5D and compute distance metrics.
    
    Returns list of dicts with all metrics.
    """
    print(f"\nScoring {len(candidates):,} candidates with Z5D...")
    results = []
    
    start_time = time.time()
    
    for i, candidate in enumerate(candidates):
        # Compute Z5D score
        n_est = z5d_n_est(str(candidate))
        z5d_score = compute_z5d_score(str(candidate), n_est)
        
        # Compute distances
        metrics = compute_distance_metrics(candidate)
        
        # Combine
        result = {
            'candidate': int(candidate),
            'z5d_score': z5d_score,
            'n_est': int(n_est),
            **metrics
        }
        results.append(result)
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(candidates) - i - 1) / rate
            print(f"  Scored {i+1:,}/{len(candidates):,} candidates "
                  f"({rate:.1f} candidates/sec, ~{remaining:.0f}s remaining)")
    
    elapsed = time.time() - start_time
    print(f"Scoring complete in {elapsed:.1f}s ({len(candidates)/elapsed:.1f} candidates/sec)")
    
    return results


def analyze_enrichment(results, top_k_slices=[100, 1000, 10000, 50000]):
    """
    Analyze spatial distribution and enrichment factors for Top-K slices.
    
    Computes:
    - Percentage of candidates within target zones (±1%, ±5% of factors)
    - Enrichment factor vs random baseline
    - Statistical summaries
    """
    # Sort by Z5D score (lower = better)
    sorted_results = sorted(results, key=lambda x: x['z5d_score'])
    
    # Baseline: entire dataset (random sample)
    total = len(results)
    baseline_near_p_1pct = sum(r['is_near_p_1pct'] for r in results) / total
    baseline_near_q_1pct = sum(r['is_near_q_1pct'] for r in results) / total
    baseline_near_any_5pct = sum(r['is_near_any_5pct'] for r in results) / total
    
    print("\n" + "="*80)
    print("ENRICHMENT ANALYSIS")
    print("="*80)
    
    print(f"\nBaseline (random uniform sampling over {total:,} candidates):")
    print(f"  Within ±1% of p: {baseline_near_p_1pct*100:.4f}%")
    print(f"  Within ±1% of q: {baseline_near_q_1pct*100:.4f}%")
    print(f"  Within ±5% of p or q: {baseline_near_any_5pct*100:.4f}%")
    
    enrichment_results = []
    
    for k in top_k_slices:
        if k > total:
            continue
            
        top_k = sorted_results[:k]
        
        pct_near_p_1pct = sum(r['is_near_p_1pct'] for r in top_k) / k
        pct_near_q_1pct = sum(r['is_near_q_1pct'] for r in top_k) / k
        pct_near_any_5pct = sum(r['is_near_any_5pct'] for r in top_k) / k
        
        # Enrichment factors (avoid division by zero)
        enrich_p_1pct = pct_near_p_1pct / baseline_near_p_1pct if baseline_near_p_1pct > 0 else float('inf')
        enrich_q_1pct = pct_near_q_1pct / baseline_near_q_1pct if baseline_near_q_1pct > 0 else float('inf')
        enrich_any_5pct = pct_near_any_5pct / baseline_near_any_5pct if baseline_near_any_5pct > 0 else float('inf')
        
        # Average distance to nearest factor
        avg_dist_nearest = sum(r['distance_to_nearest'] for r in top_k) / k
        
        # Min/max Z5D scores in this slice
        min_score = min(r['z5d_score'] for r in top_k)
        max_score = max(r['z5d_score'] for r in top_k)
        
        print(f"\nTop-{k:,} candidates:")
        print(f"  Within ±1% of p: {pct_near_p_1pct*100:.4f}% (enrichment: {enrich_p_1pct:.2f}x)")
        print(f"  Within ±1% of q: {pct_near_q_1pct*100:.4f}% (enrichment: {enrich_q_1pct:.2f}x)")
        print(f"  Within ±5% of p or q: {pct_near_any_5pct*100:.4f}% (enrichment: {enrich_any_5pct:.2f}x)")
        print(f"  Avg distance to nearest factor: {avg_dist_nearest:,.0f}")
        print(f"  Z5D score range: [{min_score:.4f}, {max_score:.4f}]")
        
        enrichment_results.append({
            'k': k,
            'pct_near_p_1pct': pct_near_p_1pct,
            'pct_near_q_1pct': pct_near_q_1pct,
            'pct_near_any_5pct': pct_near_any_5pct,
            'enrichment_p_1pct': enrich_p_1pct,
            'enrichment_q_1pct': enrich_q_1pct,
            'enrichment_any_5pct': enrich_any_5pct,
            'avg_dist_nearest': avg_dist_nearest,
            'min_z5d_score': min_score,
            'max_z5d_score': max_score
        })
    
    return {
        'baseline': {
            'near_p_1pct': baseline_near_p_1pct,
            'near_q_1pct': baseline_near_q_1pct,
            'near_any_5pct': baseline_near_any_5pct
        },
        'top_k_results': enrichment_results
    }


def save_results(results, enrichment_analysis):
    """Save results to CSV and summary to JSON."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save full results to CSV
    print(f"\nSaving results to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"Saved {len(results):,} results")
    
    # Save summary to JSON
    print(f"\nSaving summary to {SUMMARY_JSON}...")
    summary = {
        'experiment': 'Z5D Validation for N_127',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ground_truth': {
            'N_127': str(N_127),
            'p': str(TRUE_P),
            'q': str(TRUE_Q),
            'sqrt_N': str(SQRT_N),
            'p_offset_pct': -10.3902293058,
            'q_offset_pct': 11.5949736567
        },
        'parameters': {
            'num_candidates': NUM_CANDIDATES,
            'search_min': str(SEARCH_MIN),
            'search_max': str(SEARCH_MAX),
            'window_radius': str(WINDOW_RADIUS),
            'random_seed': RANDOM_SEED
        },
        'enrichment_analysis': enrichment_analysis
    }
    
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary")


def main():
    print("="*80)
    print("Z5D VALIDATION EXPERIMENT FOR N_127")
    print("="*80)
    print(f"\nGround Truth:")
    print(f"  N_127 = {N_127}")
    print(f"  p = {TRUE_P} (smaller factor)")
    print(f"  q = {TRUE_Q} (larger factor)")
    print(f"  sqrt(N) = {SQRT_N}")
    print(f"  p is at {-10.39:.2f}% below sqrt(N)")
    print(f"  q is at {+11.59:.2f}% above sqrt(N)")
    
    # Phase 1: Generate candidates
    print("\n" + "="*80)
    print("PHASE 1: GENERATE CANDIDATE DATASET")
    print("="*80)
    candidates = generate_uniform_candidates(NUM_CANDIDATES)
    
    # Phase 2: Score candidates
    print("\n" + "="*80)
    print("PHASE 2: SCORE CANDIDATES WITH Z5D")
    print("="*80)
    results = score_candidates(candidates)
    
    # Phase 3: Analyze spatial distribution
    print("\n" + "="*80)
    print("PHASE 3: ANALYZE SPATIAL DISTRIBUTION")
    print("="*80)
    enrichment_analysis = analyze_enrichment(results)
    
    # Save results
    save_results(results, enrichment_analysis)
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    top_1000 = enrichment_analysis['top_k_results'][1] if len(enrichment_analysis['top_k_results']) > 1 else None
    
    if top_1000:
        max_enrichment = max(top_1000['enrichment_p_1pct'], 
                            top_1000['enrichment_q_1pct'],
                            top_1000['enrichment_any_5pct'])
        
        print(f"\nMaximum enrichment for Top-1000: {max_enrichment:.2f}x")
        
        if max_enrichment > 5.0:
            print("\n✓ STRONG SIGNAL: Z5D provides strong guidance for factorization search")
            print("  Enrichment >5x indicates clear clustering near true factors")
        elif max_enrichment > 2.0:
            print("\n⚠ WEAK SIGNAL: Z5D shows promise but needs refinement")
            print("  Enrichment 2-5x indicates noticeable but not dramatic clustering")
        else:
            print("\n✗ NO SIGNAL: Z5D doesn't help for factorization at this scale")
            print("  Enrichment <2x indicates random distribution")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {SUMMARY_JSON}")
    print(f"\nNext steps:")
    print(f"  1. Review results in notebooks/z5d_validation_analysis.ipynb")
    print(f"  2. If promising, scale to 1M candidates")
    print(f"  3. Generate plots and statistical tests")


if __name__ == "__main__":
    main()
