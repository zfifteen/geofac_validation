#!/usr/bin/env python3
"""
p-adic vs Riemannian GVA Experiment Runner

This script compares the effectiveness of p-adic ultrametric vs traditional
Riemannian/Euclidean metrics in a toy GVA factor-finding experiment.

For each semiprime:
1. Run a simple GVA search with the baseline (Riemannian) metric
2. Run the same search with the p-adic ultrametric
3. Log results: factor found, iterations, runtime, scores

The goal is to explore whether the p-adic ultrametric provides any advantage
in identifying true factors of semiprimes.
"""

import sys
import csv
import time
import random
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import our local metrics - support both relative and direct execution
try:
    # When run as module: python -m experiments.p_adic_gva_test.src.experiment_runner
    from .metric_baseline import compute_gva_score as baseline_score
    from .metric_padic import padic_ultrametric_gva_score as padic_score
except ImportError:
    # When run directly: python experiment_runner.py
    from metric_baseline import compute_gva_score as baseline_score
    from metric_padic import padic_ultrametric_gva_score as padic_score


# Toy semiprimes for testing (RSA-100 to RSA-200 range)
# We'll use some smaller ones for quick testing, plus a few from RSA challenges
SEMIPRIMES = [
    {
        "name": "Toy-1",
        "N": 143,  # 11 × 13
        "p": 11,
        "q": 13,
        "description": "Minimal test case"
    },
    {
        "name": "Toy-2", 
        "N": 1763,  # 41 × 43
        "p": 41,
        "q": 43,
        "description": "Small twin-prime product"
    },
    {
        "name": "Toy-3",
        "N": 6557,  # 79 × 83
        "p": 79,
        "q": 83,
        "description": "Small twin-prime product"
    },
    {
        "name": "Medium-1",
        "N": 9746347772161,  # 3122977 × 3122987 (22-bit factors)
        "p": 3122977,
        "q": 3122987,
        "description": "6-digit factors"
    },
    {
        "name": "RSA-100-mini",
        "N": int("1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139"),
        "p": int("37975227936943673922808872755445627854565536638199"),
        "q": int("40094690950920881030683735292761468389214899724061"),
        "description": "Actual RSA-100 challenge"
    },
]


def integer_sqrt(n: int) -> int:
    """Compute integer square root of n."""
    if n < 0:
        raise ValueError("Cannot compute sqrt of negative number")
    if n < 2:
        return n
    
    # Newton's method for integer sqrt
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def generate_search_candidates(N: int, sqrt_N: int, num_candidates: int, 
                               window_pct: float = 15.0, seed: int = None) -> List[int]:
    """
    Generate candidate factors uniformly in a window around sqrt(N).
    
    Args:
        N: The semiprime
        sqrt_N: Integer square root of N
        num_candidates: Number of candidates to generate
        window_pct: Window size as percentage of sqrt_N (default 15%)
        seed: Random seed for reproducibility
        
    Returns:
        List of candidate integers (odd only, since we assume odd semiprime)
    """
    if seed is not None:
        random.seed(seed)
    
    # Define search window
    window_radius = int(sqrt_N * window_pct / 100)
    search_min = max(3, sqrt_N - window_radius)
    search_max = sqrt_N + window_radius
    
    candidates = []
    for _ in range(num_candidates):
        # Generate random candidate in window
        cand = random.randint(search_min, search_max)
        
        # Make it odd (since N is odd, factors must be odd)
        if cand % 2 == 0:
            cand += 1
        
        # Keep in bounds
        if cand > search_max:
            cand = search_max if search_max % 2 == 1 else search_max - 1
        
        candidates.append(cand)
    
    return candidates


def gcd(a: int, b: int) -> int:
    """Compute GCD using Euclidean algorithm."""
    return math.gcd(a, b)


def run_gva_search(N: int, sqrt_N: int, metric_name: str, 
                   score_func, num_candidates: int = 500,
                   window_pct: float = 15.0, seed: int = None) -> Dict:
    """
    Run a simple GVA-style factor search using the given metric.
    
    Strategy:
    1. Generate candidates uniformly in window around sqrt(N)
    2. Score each candidate with the given metric
    3. Sort by score (lower is better)
    4. Test top candidates with GCD
    5. Stop if factor found
    
    Args:
        N: Semiprime to factor
        sqrt_N: Integer sqrt of N
        metric_name: Name of metric ("baseline" or "padic")
        score_func: Scoring function to use
        num_candidates: Number of candidates to generate
        window_pct: Search window size
        seed: Random seed
        
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    # Generate candidates
    candidates = generate_search_candidates(N, sqrt_N, num_candidates, window_pct, seed)
    
    # Score all candidates
    scored_candidates = []
    for cand in candidates:
        try:
            if metric_name == "padic":
                score = score_func(cand, sqrt_N, N)
            else:
                score = score_func(cand, sqrt_N)
            scored_candidates.append((cand, score))
        except Exception as e:
            # Skip candidates that cause errors
            continue
    
    # Sort by score (lower is better)
    scored_candidates.sort(key=lambda x: x[1])
    
    # Try top candidates with GCD
    factor_found = None
    iterations_to_factor = 0
    gcd_checks = 0
    
    for i, (cand, score) in enumerate(scored_candidates):
        gcd_checks += 1
        g = gcd(cand, N)
        
        if g > 1 and g < N:
            # Found a nontrivial factor!
            factor_found = g
            iterations_to_factor = i + 1
            break
    
    elapsed_time = time.time() - start_time
    
    # Compute alignment score (how well-ranked was the best candidate?)
    best_score = scored_candidates[0][1] if scored_candidates else None
    worst_score = scored_candidates[-1][1] if scored_candidates else None
    
    result = {
        "metric": metric_name,
        "num_candidates": num_candidates,
        "window_pct": window_pct,
        "factor_found": factor_found is not None,
        "factor_value": factor_found,
        "iterations_to_factor": iterations_to_factor if factor_found else None,
        "gcd_checks": gcd_checks,
        "runtime_seconds": elapsed_time,
        "best_score": best_score,
        "worst_score": worst_score,
        "total_scored": len(scored_candidates)
    }
    
    return result


def run_experiment_on_semiprime(semiprime_data: Dict, num_candidates: int = 500,
                               window_pct: float = 15.0, seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Run complete experiment on one semiprime: both metrics.
    
    Args:
        semiprime_data: Dict with N, p, q, name, description
        num_candidates: Number of candidates per search
        window_pct: Search window size
        seed: Random seed
        
    Returns:
        (baseline_result, padic_result)
    """
    N = semiprime_data["N"]
    sqrt_N = integer_sqrt(N)
    
    print(f"\n{'='*80}")
    print(f"Testing: {semiprime_data['name']}")
    print(f"N = {N}")
    print(f"p = {semiprime_data['p']}, q = {semiprime_data['q']}")
    print(f"sqrt(N) ≈ {sqrt_N}")
    print(f"{'='*80}")
    
    # Run with baseline metric
    print(f"\n[1/2] Running with BASELINE (Riemannian/Z5D) metric...")
    baseline_result = run_gva_search(
        N, sqrt_N, "baseline", baseline_score,
        num_candidates, window_pct, seed
    )
    print(f"  Factor found: {baseline_result['factor_found']}")
    if baseline_result['factor_found']:
        print(f"  Factor: {baseline_result['factor_value']}")
        print(f"  Iterations: {baseline_result['iterations_to_factor']}")
    print(f"  Runtime: {baseline_result['runtime_seconds']:.4f}s")
    
    # Run with p-adic metric
    print(f"\n[2/2] Running with P-ADIC ultrametric...")
    padic_result = run_gva_search(
        N, sqrt_N, "padic", padic_score,
        num_candidates, window_pct, seed
    )
    print(f"  Factor found: {padic_result['factor_found']}")
    if padic_result['factor_found']:
        print(f"  Factor: {padic_result['factor_value']}")
        print(f"  Iterations: {padic_result['iterations_to_factor']}")
    print(f"  Runtime: {padic_result['runtime_seconds']:.4f}s")
    
    return baseline_result, padic_result


def save_results_to_csv(results: List[Dict], output_path: Path):
    """Save experiment results to CSV."""
    if not results:
        print("No results to save")
        return
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all field names
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())
    fieldnames = sorted(fieldnames)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main experiment runner."""
    print("="*80)
    print("p-adic vs Riemannian GVA Factor-Finding Experiment")
    print("="*80)
    
    # Experiment parameters
    NUM_CANDIDATES = 500  # Number of candidates per search
    WINDOW_PCT = 15.0     # Search window: ±15% around sqrt(N)
    SEED = 42             # Random seed for reproducibility
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    all_results = []
    
    for semiprime_data in SEMIPRIMES:
        try:
            baseline_result, padic_result = run_experiment_on_semiprime(
                semiprime_data, NUM_CANDIDATES, WINDOW_PCT, SEED
            )
            
            # Add semiprime metadata to results
            for result in [baseline_result, padic_result]:
                result["semiprime_name"] = semiprime_data["name"]
                result["N"] = str(semiprime_data["N"])  # Store as string for large numbers
                result["true_p"] = str(semiprime_data["p"])
                result["true_q"] = str(semiprime_data["q"])
                result["description"] = semiprime_data["description"]
            
            all_results.append(baseline_result)
            all_results.append(padic_result)
            
        except Exception as e:
            print(f"\nERROR processing {semiprime_data['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = output_dir / f"padic_gva_results_{timestamp}.csv"
    save_results_to_csv(all_results, output_csv)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    baseline_successes = sum(1 for r in all_results if r["metric"] == "baseline" and r["factor_found"])
    padic_successes = sum(1 for r in all_results if r["metric"] == "padic" and r["factor_found"])
    total_semiprimes = len(SEMIPRIMES)
    
    print(f"\nTotal semiprimes tested: {total_semiprimes}")
    print(f"Baseline metric successes: {baseline_successes}/{total_semiprimes}")
    print(f"p-adic metric successes: {padic_successes}/{total_semiprimes}")
    
    # Compare performance where both found factors
    print("\n" + "-"*80)
    print("Detailed comparison:")
    print("-"*80)
    print(f"{'Semiprime':<20} {'Baseline':<15} {'p-adic':<15} {'Winner':<15}")
    print("-"*80)
    
    for i in range(0, len(all_results), 2):
        baseline = all_results[i]
        padic = all_results[i+1] if i+1 < len(all_results) else None
        
        if padic is None:
            continue
        
        name = baseline["semiprime_name"]
        baseline_iters = baseline["iterations_to_factor"] if baseline["factor_found"] else "Failed"
        padic_iters = padic["iterations_to_factor"] if padic["factor_found"] else "Failed"
        
        # Determine winner
        if baseline["factor_found"] and not padic["factor_found"]:
            winner = "Baseline"
        elif padic["factor_found"] and not baseline["factor_found"]:
            winner = "p-adic"
        elif baseline["factor_found"] and padic["factor_found"]:
            if baseline["iterations_to_factor"] < padic["iterations_to_factor"]:
                winner = "Baseline"
            elif padic["iterations_to_factor"] < baseline["iterations_to_factor"]:
                winner = "p-adic"
            else:
                winner = "Tie"
        else:
            winner = "Both failed"
        
        print(f"{name:<20} {str(baseline_iters):<15} {str(padic_iters):<15} {winner:<15}")
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
