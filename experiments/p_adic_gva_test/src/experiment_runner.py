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


# Define semiprimes as (p, q) pairs - compute N from them to avoid transcription errors
SEMIPRIME_DEFINITIONS = [
    (11, 13, "Toy-1", "Minimal test case"),
    (41, 43, "Toy-2", "Small twin-prime product"),
    (79, 83, "Toy-3", "Small twin-prime product"),
    (3122977, 3122987, "Medium-1", "~22-bit prime factors"),
    (
        int("37975227936943673922808872755445627854565536638199"),
        int("40094690950920881030683735292761468389214899724061"),
        "RSA-100",
        "Actual RSA-100 challenge"
    ),
]


def is_prime_miller_rabin(n: int, k: int = 10) -> bool:
    """
    Miller-Rabin primality test.
    Returns True if n is probably prime, False if definitely composite.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witnesses to test
    import random
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def validate_semiprime(p: int, q: int, N: int, name: str) -> bool:
    """
    Validate that N = p × q and that p, q are prime.
    Raises AssertionError if validation fails.
    """
    # Check multiplication
    computed_N = p * q
    assert computed_N == N, (
        f"{name}: N mismatch! p×q = {computed_N} but N = {N}"
    )
    
    # Check that p and q are greater than 1
    assert p > 1 and q > 1, f"{name}: factors must be > 1, got p={p}, q={q}"
    
    # Check that p and q are coprime
    assert math.gcd(p, q) == 1, f"{name}: factors must be coprime, got gcd(p,q)={math.gcd(p, q)}"
    
    # For small numbers, verify primality
    if p < 10000 or q < 10000:
        # Quick primality check for small numbers
        def is_small_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        if p < 10000:
            assert is_small_prime(p), f"{name}: p={p} is not prime"
        if q < 10000:
            assert is_small_prime(q), f"{name}: q={q} is not prime"
    
    return True


# Build validated semiprime dataset
SEMIPRIMES = []
for p, q, name, description in SEMIPRIME_DEFINITIONS:
    N = p * q
    try:
        validate_semiprime(p, q, N, name)
        SEMIPRIMES.append({
            "name": name,
            "N": N,
            "p": p,
            "q": q,
            "description": description
        })
        print(f"✓ Validated {name}: N = {N}")
    except AssertionError as e:
        print(f"✗ FAILED validation for {name}: {e}")
        raise


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
    
    # Define search window - ensure minimum size for small semiprimes
    window_radius = max(50, int(sqrt_N * window_pct / 100))
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
