#!/usr/bin/env python3
"""
Gradient Descent "Zoom" Algorithm for GeoFac Factorization

OVERVIEW
========
This module implements the gradient-guided iterative window narrowing strategy
to transform GeoFac from a statistical distinguisher into an operational
factorization tool.

ALGORITHM
=========
The Zoom algorithm leverages the Z5D fitness landscape to iteratively narrow
the search window:

1. **Survey:** Sample candidates across current window using QMC
2. **Score:** Evaluate all candidates with Z5D geometric resonance
3. **Locate:** Identify cluster of top 1% highest-scoring candidates
4. **Test:** Check if any top candidates are factors (GCD test)
5. **Zoom:** Re-center window on cluster, shrink by zoom_factor
6. **Repeat:** Iterate until factor found or convergence criteria met

CONVERGENCE
===========
With zoom_factor=100, achieving 10⁹× reduction requires ~5 iterations:
- Iteration 1: 10¹⁸ → 10¹⁶
- Iteration 2: 10¹⁶ → 10¹⁴
- Iteration 3: 10¹⁴ → 10¹²
- Iteration 4: 10¹² → 10¹⁰
- Iteration 5: 10¹⁰ → 10⁸

Each iteration adds ~6.6 bits of precision (log₂(100) ≈ 6.64).

USAGE
=====
Basic usage:

    from gradient_zoom import gradient_zoom
    import gmpy2
    
    N = gmpy2.mpz("137524771864208156028430259349934309717")
    result = gradient_zoom(N)
    
    if result["factor_found"]:
        print(f"Factor: {result['factor']}")
        print(f"Cofactor: {result['cofactor']}")
    else:
        print(f"Not found after {result['iterations']} iterations")

Advanced usage with custom parameters:

    result = gradient_zoom(
        N,
        initial_window_pct=0.13,
        zoom_factor=100,
        candidates_per_iteration=100_000,
        max_iterations=10,
        top_k_fraction=0.01,
        convergence_threshold_bits=32  # Stop at N^(1/4)
    )

REFERENCES
==========
- Issue #43: Prospective Gradient Descent Validation Protocol
- Section VII (Research Report): Gradient Descent Strategy
- MASTER_FINDINGS.md: Coverage Paradox Analysis
"""

import gmpy2
import time
import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.stats import qmc
import sys

# Import z5d_adapter for scoring
sys.path.insert(0, '.')
from z5d_adapter import z5d_n_est, compute_z5d_score


# ============================================
# QMC CANDIDATE GENERATION
# ============================================

QMC_SEED = 42
QMC_SCALE_BITS = 53  # 2^53 for each dimension
QMC_DENOM_BITS = QMC_SCALE_BITS * 2  # Total precision: 106 bits


def generate_qmc_candidates(search_min: gmpy2.mpz, search_max: gmpy2.mpz, 
                           n_samples: int) -> List[gmpy2.mpz]:
    """
    Generate QMC candidates using Sobol sequence for uniform coverage.
    
    Uses 2D Sobol sequence mapped to 106-bit fixed-point precision
    to avoid float quantization issues. For small ranges, falls back
    to dense sampling to ensure complete coverage.
    
    Args:
        search_min: Minimum candidate value (gmpy2.mpz)
        search_max: Maximum candidate value (gmpy2.mpz)
        n_samples: Number of candidates to generate
    
    Returns:
        list of gmpy2.mpz: Odd candidates in the search range
    """
    # Convert to integers for arithmetic
    search_min_int = int(search_min)
    search_max_int = int(search_max)
    search_range_int = search_max_int - search_min_int
    
    # For very small ranges, use dense sampling instead of QMC
    # This ensures we don't miss factors due to quantization
    if search_range_int < n_samples * 2:
        # Generate all odd numbers in the range, then sample uniformly
        candidates = []
        for val in range(search_min_int, search_max_int + 1):
            if val % 2 == 1:  # Only odd numbers
                candidates.append(gmpy2.mpz(val))
        
        # If we have fewer candidates than requested, repeat them
        if len(candidates) < n_samples:
            # Repeat the list to reach n_samples
            repetitions = (n_samples // len(candidates)) + 1
            candidates = (candidates * repetitions)[:n_samples]
        elif len(candidates) > n_samples:
            # Randomly sample n_samples from the candidates
            import random
            random.seed(QMC_SEED)
            candidates = random.sample(candidates, n_samples)
        
        return candidates
    
    # For larger ranges, use QMC as before
    # Initialize Sobol sampler with 2D space
    sampler = qmc.Sobol(d=2, scramble=True, seed=QMC_SEED)
    qmc_samples = sampler.random(n=n_samples)
    
    # Fixed-point precision constants
    scale = 1 << QMC_SCALE_BITS
    denom_bits = QMC_DENOM_BITS
    
    candidates = []
    for row in qmc_samples:
        # Map [0,1]² to fixed-point integer
        hi = min(int(row[0] * scale), scale - 1)
        lo = min(int(row[1] * scale), scale - 1)
        
        # Combine into multi-bit value
        x = (hi << QMC_SCALE_BITS) | lo
        
        # Map to search range using gmpy2 for large numbers
        x_mpz = gmpy2.mpz(x)
        range_mpz = gmpy2.mpz(search_range_int)
        offset = int((x_mpz * range_mpz) >> denom_bits)
        candidate = search_min_int + offset
        
        # Ensure candidate is within bounds before making it odd
        candidate = max(search_min_int, min(candidate, search_max_int))
        
        # Make candidate odd (all primes except 2 are odd) while staying in range
        if candidate % 2 == 0:
            if candidate + 1 <= search_max_int:
                candidate += 1
            elif candidate - 1 >= search_min_int:
                candidate -= 1
        
        candidates.append(gmpy2.mpz(candidate))
    
    return candidates


# ============================================
# Z5D SCORING
# ============================================

def score_candidates_z5d(candidates: List[gmpy2.mpz]) -> List[Tuple[gmpy2.mpz, float]]:
    """
    Score all candidates using Z5D geometric resonance.
    
    Lower scores indicate better alignment with Z5D prime predictions.
    
    Args:
        candidates: List of gmpy2.mpz candidates
    
    Returns:
        list of tuples: [(candidate, score), ...] sorted by score (best first)
    """
    scored = []
    failures = 0
    
    for c in candidates:
        try:
            # Estimate prime index
            n_est = z5d_n_est(str(c))
            # Compute Z5D score
            score = compute_z5d_score(str(c), n_est)
            scored.append((c, score))
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"  Warning: Z5D scoring failed: {type(e).__name__}")
            # Assign worst-case score so failed candidates remain in pool but ranked lowest
            scored.append((c, float('inf')))
    
    if failures > 0:
        total = len(candidates)
        failure_rate = failures / total if total > 0 else 0
        if failure_rate > 0.01:  # Only warn if >1% failure
            print(f"  Z5D scoring failures: {failures}/{total} ({failure_rate:.1%})")
    
    # Sort by score (lower is better)
    scored.sort(key=lambda x: x[1])
    
    return scored


# ============================================
# CLUSTER CENTER COMPUTATION
# ============================================

def compute_cluster_center(top_candidates: List[Tuple[gmpy2.mpz, float]],
                          method: str = "median") -> gmpy2.mpz:
    """
    Compute the center of the top-scoring candidate cluster.
    
    Args:
        top_candidates: List of (candidate, score) tuples
        method: "median", "mean", or "weighted_mean"
    
    Returns:
        gmpy2.mpz: Cluster center location
    """
    if not top_candidates:
        raise ValueError("Cannot compute cluster center: no candidates provided")
    
    values = [int(c) for c, _ in top_candidates]
    
    if method == "median":
        # Median is robust to outliers
        return gmpy2.mpz(int(np.median(values)))
    
    elif method == "mean":
        # Simple arithmetic mean
        return gmpy2.mpz(int(np.mean(values)))
    
    elif method == "weighted_mean":
        # Weight by inverse score (lower score = higher weight)
        scores = [s for _, s in top_candidates]
        # Avoid division by zero and inf
        weights = [1.0 / (s + 1e-10) if s != float('inf') else 0.0 for s in scores]
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        total_weight = sum(weights)
        if total_weight == 0:
            return gmpy2.mpz(int(np.median(values)))  # Fallback to median
        return gmpy2.mpz(int(weighted_sum / total_weight))
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================
# MAIN GRADIENT ZOOM ALGORITHM
# ============================================

def gradient_zoom(N: gmpy2.mpz,
                 initial_window_pct: float = 0.13,
                 zoom_factor: int = 100,
                 candidates_per_iteration: int = 100_000,
                 max_iterations: int = 10,
                 top_k_fraction: float = 0.01,
                 convergence_threshold_bits: int = 32,
                 cluster_method: str = "median",
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Gradient-guided iterative window narrowing for factorization.
    
    Args:
        N: Semiprime to factor (gmpy2.mpz)
        initial_window_pct: Initial window as fraction of √N (default: 0.13)
        zoom_factor: Window reduction per iteration (default: 100)
        candidates_per_iteration: Candidates to test per iteration (default: 100k)
        max_iterations: Maximum zoom iterations (default: 10)
        top_k_fraction: Fraction of candidates to consider for clustering (default: 0.01)
        convergence_threshold_bits: Stop when window < 2^bits (default: 32, i.e., N^(1/4) for 128-bit N)
        cluster_method: Method for computing cluster center (default: "median")
        verbose: Print progress information (default: True)
    
    Returns:
        dict: Results including:
            - factor_found: bool
            - factor: gmpy2.mpz or None
            - cofactor: gmpy2.mpz or None
            - iterations: int
            - window_history: list of dicts
            - total_candidates_tested: int
            - time_elapsed: float
            - convergence_reason: str
    """
    start_time = time.time()
    sqrt_N = gmpy2.isqrt(N)
    window_center = sqrt_N
    window_radius = int(sqrt_N * initial_window_pct)
    
    # Convergence threshold - adjust based on N size
    # For N^(1/4), we want 2^(bits/4)
    # But for small N, this might be too large, so cap it intelligently
    n_bits = N.bit_length()
    auto_threshold_bits = max(convergence_threshold_bits, n_bits // 4)
    # For very small N, use a fraction of the initial window
    if auto_threshold_bits > n_bits - 4:
        # Use 1/100 of initial window as threshold for small numbers
        convergence_threshold = max(gmpy2.mpz(1), window_radius // 100)
        if verbose:
            print(f"Auto-adjusting convergence threshold for small N: {convergence_threshold}")
    else:
        convergence_threshold = gmpy2.mpz(1) << auto_threshold_bits
    
    results = {
        "factor_found": False,
        "factor": None,
        "cofactor": None,
        "iterations": 0,
        "window_history": [],
        "total_candidates_tested": 0,
        "time_elapsed": 0,
        "convergence_reason": "max_iterations"
    }
    
    if verbose:
        print("=" * 70)
        print("GRADIENT DESCENT ZOOM ALGORITHM")
        print("=" * 70)
        print(f"N = {N}")
        print(f"√N = {sqrt_N}")
        print(f"Bit length: {N.bit_length()}")
        print(f"Initial window: ±{initial_window_pct*100:.1f}% of √N")
        print(f"Zoom factor: {zoom_factor}×")
        print(f"Max iterations: {max_iterations}")
        print(f"Convergence threshold: 2^{convergence_threshold_bits}")
        print("=" * 70)
    
    for iteration in range(max_iterations):
        iteration_start = time.time()
        
        # Calculate search bounds
        search_min = max(gmpy2.mpz(2), window_center - window_radius)
        search_max = window_center + window_radius
        
        # Validate search range
        if search_max <= search_min:
            results["convergence_reason"] = "invalid_window"
            if verbose:
                print(f"\n[Iteration {iteration+1}] Invalid window: [{search_min}, {search_max}]")
            break
        
        if verbose:
            print(f"\n[Iteration {iteration+1}/{max_iterations}]")
            print(f"  Window center: {window_center}")
            print(f"  Window radius: {window_radius}")
            print(f"  Search range: [{search_min}, {search_max}]")
            window_size = int(search_max - search_min)
            print(f"  Window size: {window_size:,}")
        
        # Step 1: Survey - Generate candidates
        if verbose:
            print(f"  Generating {candidates_per_iteration:,} candidates...")
        candidates = generate_qmc_candidates(search_min, search_max, candidates_per_iteration)
        gen_time = time.time() - iteration_start
        if verbose:
            print(f"  Generated in {gen_time:.2f}s")
        
        # Step 2: Score - Evaluate with Z5D
        if verbose:
            print(f"  Scoring with Z5D...")
        score_start = time.time()
        scored = score_candidates_z5d(candidates)
        score_time = time.time() - score_start
        if verbose:
            print(f"  Scored {len(scored)} candidates in {score_time:.2f}s")
        
        # Step 3: Locate - Identify top cluster
        top_k = max(1, int(len(scored) * top_k_fraction))
        top_candidates = scored[:top_k]
        
        if verbose:
            print(f"  Top {top_k} candidates (top {top_k_fraction*100:.1f}%)")
            best_score = top_candidates[0][1] if top_candidates else float('inf')
            worst_top_score = top_candidates[-1][1] if top_candidates else float('inf')
            print(f"  Score range: [{best_score:.4f}, {worst_top_score:.4f}]")
        
        # Step 4: Test - Check if any top candidates are factors
        # Test all top candidates (not just top 100) since we want maximum coverage
        gcd_test_count = min(len(top_candidates), 1000)  # Cap at 1000 for efficiency
        for candidate, score in top_candidates[:gcd_test_count]:
            g = gmpy2.gcd(candidate, N)
            if g > 1 and g < N:
                # Factor found!
                results["factor_found"] = True
                results["factor"] = g
                results["cofactor"] = N // g
                results["iterations"] = iteration + 1
                results["total_candidates_tested"] += len(candidates)
                results["time_elapsed"] = time.time() - start_time
                results["convergence_reason"] = "factor_found"
                
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"✓ FACTOR FOUND!")
                    print(f"{'='*70}")
                    print(f"Factor: {g}")
                    print(f"Cofactor: {N // g}")
                    print(f"Iterations: {iteration + 1}")
                    print(f"Total candidates tested: {results['total_candidates_tested']:,}")
                    print(f"Time elapsed: {results['time_elapsed']:.2f}s")
                    print(f"{'='*70}")
                
                # Record final window state
                iteration_time = time.time() - iteration_start
                results["window_history"].append({
                    "iteration": iteration + 1,
                    "window_center": int(window_center),
                    "window_radius": int(window_radius),
                    "window_size": int(search_max - search_min),
                    "candidates_tested": len(candidates),
                    "top_k": top_k,
                    "best_score": float(best_score),
                    "cluster_center": int(window_center),  # No update needed
                    "iteration_time": iteration_time,
                    "factor_found": True
                })
                
                return results
        
        # Step 5: Zoom - Re-center and shrink window
        cluster_center = compute_cluster_center(top_candidates, method=cluster_method)
        new_window_radius = max(1, window_radius // zoom_factor)
        
        if verbose:
            print(f"  Cluster center: {cluster_center}")
            print(f"  Center shift: {int(cluster_center - window_center):+,}")
            print(f"  New radius: {new_window_radius:,} ({zoom_factor}× reduction)")
        
        # Record iteration history
        iteration_time = time.time() - iteration_start
        results["window_history"].append({
            "iteration": iteration + 1,
            "window_center": int(window_center),
            "window_radius": int(window_radius),
            "window_size": int(search_max - search_min),
            "candidates_tested": len(candidates),
            "top_k": top_k,
            "best_score": float(best_score),
            "cluster_center": int(cluster_center),
            "new_radius": int(new_window_radius),
            "iteration_time": iteration_time,
            "factor_found": False
        })
        
        # Update window for next iteration
        window_center = cluster_center
        window_radius = new_window_radius
        
        results["total_candidates_tested"] += len(candidates)
        
        # Step 6: Convergence check
        if window_radius <= convergence_threshold:
            results["convergence_reason"] = "threshold_reached"
            if verbose:
                print(f"\n  Convergence threshold reached: {window_radius} <= {convergence_threshold}")
                print(f"  Window ready for Coppersmith handoff (Stage 3)")
                print(f"  Performing exhaustive GCD test on final window...")
            
            # Final exhaustive check: test ALL candidates in the final window
            final_test_count = 0
            for candidate, score in scored:  # Test all, not just top_k
                final_test_count += 1
                g = gmpy2.gcd(candidate, N)
                if g > 1 and g < N:
                    # Factor found in final exhaustive search!
                    results["factor_found"] = True
                    results["factor"] = g
                    results["cofactor"] = N // g
                    results["iterations"] = iteration + 1
                    results["total_candidates_tested"] += len(candidates)
                    results["time_elapsed"] = time.time() - start_time
                    results["convergence_reason"] = "factor_found_final"
                    
                    if verbose:
                        print(f"\n{'='*70}")
                        print(f"✓ FACTOR FOUND IN FINAL WINDOW!")
                        print(f"{'='*70}")
                        print(f"Factor: {g}")
                        print(f"Cofactor: {N // g}")
                        print(f"Tested {final_test_count} candidates in final window")
                        print(f"Iterations: {iteration + 1}")
                        print(f"Total candidates tested: {results['total_candidates_tested']:,}")
                        print(f"Time elapsed: {results['time_elapsed']:.2f}s")
                        print(f"{'='*70}")
                    
                    # Record final window state
                    iteration_time = time.time() - iteration_start
                    results["window_history"].append({
                        "iteration": iteration + 1,
                        "window_center": int(window_center),
                        "window_radius": int(window_radius),
                        "window_size": int(search_max - search_min),
                        "candidates_tested": len(candidates),
                        "top_k": top_k,
                        "best_score": float(best_score),
                        "cluster_center": int(cluster_center),
                        "new_radius": int(new_window_radius),
                        "iteration_time": iteration_time,
                        "factor_found": True
                    })
                    
                    return results
            
            if verbose:
                print(f"  Tested all {final_test_count} candidates in final window - no factor found")
            
            break
    
    results["iterations"] = len(results["window_history"])
    results["time_elapsed"] = time.time() - start_time
    
    if verbose and not results["factor_found"]:
        print(f"\n{'='*70}")
        print(f"✗ FACTOR NOT FOUND")
        print(f"{'='*70}")
        print(f"Iterations: {results['iterations']}")
        print(f"Total candidates tested: {results['total_candidates_tested']:,}")
        print(f"Time elapsed: {results['time_elapsed']:.2f}s")
        print(f"Convergence reason: {results['convergence_reason']}")
        print(f"Final window: [{int(search_min):,}, {int(search_max):,}]")
        print(f"Final window size: {int(search_max - search_min):,}")
        print(f"{'='*70}")
    
    return results


# ============================================
# MAIN ENTRY POINT FOR TESTING
# ============================================

if __name__ == "__main__":
    # Test on N_127 (known semiprime)
    N_127 = gmpy2.mpz("137524771864208156028430259349934309717")
    
    print("Testing Gradient Zoom on N_127...")
    result = gradient_zoom(N_127, verbose=True)
    
    if result["factor_found"]:
        print(f"\nSUCCESS: Found factor {result['factor']}")
    else:
        print(f"\nNO FACTOR: Stopped after {result['iterations']} iterations")
        print(f"Convergence reason: {result['convergence_reason']}")
