#!/usr/bin/env python3
"""
Improved Experiment Design: run_adaptive_blind_factor.py

OVERVIEW
========
This script implements a blind factorization algorithm using adaptive window
search with Z5D geometric resonance scoring. It tests whether we can factor
semiprimes without prior knowledge of factor locations by iteratively expanding
the search window until factors are found.

KEY FEATURES
============
1. **Adaptive Window Strategy**: Tests progressively larger windows (13%, 20%, 
   30%, 50%, 75%, 100%, 150%, 200%, 300% of sqrt(N)) until factor found or 
   timeout reached.

2. **Asymmetric Windowing**: Based on observed q-enrichment patterns, searches 
   30% below sqrt(N) and 100% above sqrt(N) to bias toward larger factors.

3. **QMC Candidate Generation**: Uses Sobol quasi-Monte Carlo sequences with 
   106-bit fixed-point precision for uniform coverage without float quantization.

4. **Z5D Geometric Resonance Scoring**: Scores candidates using prime number 
   theorem predictions via the z5d_adapter module.

5. **Blind Testing with Verification**: Uses known test semiprimes where factors
   are "hidden" during search but available for post-hoc verification.

USAGE
=====
Run with default parameters (1 hour timeout, 500k candidates per window):
    python3 run_adaptive_blind_factor.py

Customize parameters in code:
    CANDIDATES_PER_WINDOW = 100_000  # Reduce for faster testing
    WINDOW_SEQUENCE = [0.13, 0.20]   # Test fewer windows
    MAX_WALLCLOCK_SECONDS = 600      # 10 minute timeout

OUTPUT
======
Results are saved to: adaptive_blind_results.json

Format:
{
  "N_127": {
    "factor_found": true/false,
    "factor": <factor if found>,
    "cofactor": <cofactor if found>,
    "window_pct": <window where found>,
    "windows_tested": [<detailed window stats>],
    "time_elapsed": <seconds>,
    "verification": "CORRECT"/"INCORRECT"/"NOT_FOUND"
  }
}

IMPLEMENTATION NOTES
====================
- All arithmetic uses gmpy2.mpz for arbitrary precision (no np.int64)
- QMC mapping uses gmpy2 to avoid overflow on large search ranges
- Search windows are clamped to stay positive (min value = 2)
- JSON serialization recursively converts gmpy2.mpz to strings

REFERENCES
==========
- Based on methodology from adversarial_test_adaptive.py
- Uses z5d_adapter.py for geometric resonance scoring
- Follows repository guidelines for avoiding fixed-width integers
"""

import gmpy2
import numpy as np
import time
import json
from pathlib import Path
from scipy.stats import qmc

# Add current directory to path for z5d_adapter imports
import sys
sys.path.insert(0, '.')
from z5d_adapter import z5d_n_est, compute_z5d_score

# ============================================
# EXPERIMENT PARAMETERS - COLLABORATIVE DESIGN
# ============================================

# Target: Use a semiprime where we KNOW the factors but PRETEND we don't
# This allows verification while testing blind methodology
TEST_SEMIPRIMES = {
    "N_127": {
        "N": gmpy2.mpz("137524771864208156028430259349934309717"),
        "p_true": gmpy2.mpz("10508623501177419659"),  # HIDDEN during search
        "q_true": gmpy2.mpz("13086849276577416863"),  # HIDDEN during search
        "bits": 127
    },
    # Add more test cases for validation
}

# Adaptive window sequence (as fraction of sqrt(N))
WINDOW_SEQUENCE = [0.13, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]

# Extended timeout for thorough testing
MAX_WALLCLOCK_SECONDS = 3600  # 1 hour

# Enrichment detection threshold (stop if signal detected)
ENRICHMENT_THRESHOLD = 5.0  # 5× enrichment triggers deeper search

# Candidates per window
CANDIDATES_PER_WINDOW = 500_000
TOP_K_FRACTION = 0.01  # Test top 1%

# QMC generation constants
QMC_SEED = 42  # Seed for reproducibility
QMC_SCALE_BITS = 53  # 2^53 for each dimension
QMC_DENOM_BITS = 106  # Total precision: 53 + 53 bits

# Asymmetric window biasing (based on observed q-enrichment)
WINDOW_BIAS_BELOW = 0.3  # 30% below sqrt(N)
WINDOW_BIAS_ABOVE = 1.0  # 100% above sqrt(N)


def generate_qmc_candidates(search_min, search_max, n_samples):
    """
    Generate QMC candidates using Sobol sequence for uniform coverage.
    
    Uses 2D Sobol sequence mapped to 106-bit fixed-point precision
    to avoid float quantization issues.
    
    Args:
        search_min: Minimum candidate value (gmpy2.mpz or int)
        search_max: Maximum candidate value (gmpy2.mpz or int)
        n_samples: Number of candidates to generate
    
    Returns:
        list of gmpy2.mpz: Odd candidates in the search range
    """
    # Initialize Sobol sampler with 2D space
    sampler = qmc.Sobol(d=2, scramble=True, seed=QMC_SEED)
    qmc_samples = sampler.random(n=n_samples)
    
    # Convert to integers for arithmetic
    search_min_int = int(search_min)
    search_max_int = int(search_max)
    search_range_int = search_max_int - search_min_int
    
    # Fixed-point precision constants
    scale = 1 << QMC_SCALE_BITS
    denom_bits = QMC_DENOM_BITS
    
    candidates = []
    for row in qmc_samples:
        # Map [0,1]² to fixed-point integer
        hi = min(int(row[0] * scale), scale - 1)
        lo = min(int(row[1] * scale), scale - 1)
        
        # Combine into multi-bit value
        # NOTE: This assumes QMC_SCALE_BITS = 53 for proper 106-bit precision
        # If QMC_SCALE_BITS is changed, QMC_DENOM_BITS must also be updated accordingly
        assert QMC_SCALE_BITS == 53, "QMC_SCALE_BITS must be 53 for current implementation"
        x = (hi << QMC_SCALE_BITS) | lo
        
        # Map to search range using gmpy2 for large numbers
        x_mpz = gmpy2.mpz(x)
        range_mpz = gmpy2.mpz(search_range_int + 1)
        offset = int((x_mpz * range_mpz) >> denom_bits)
        candidate = search_min_int + offset
        
        # Make candidate odd (all primes except 2 are odd)
        if candidate % 2 == 0:
            candidate += 1
            if candidate > search_max_int:
                candidate -= 2
        
        candidates.append(gmpy2.mpz(candidate))
    
    return candidates


def score_candidates_z5d(candidates):
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
    
    if failures > 0:
        print(f"  Total scoring failures: {failures}/{len(candidates)}")
    
    # Sort by score (lower is better)
    scored.sort(key=lambda x: x[1])
    
    return scored


def run_adaptive_window_search(N: gmpy2.mpz, max_time: int) -> dict:
    """
    Attempt blind factorization using adaptive window strategy.
    
    Key insight from analysis: The q-enrichment suggests searching
    ABOVE sqrt(N) may be more productive than symmetric search.
    
    Args:
        N: The semiprime to factor
        max_time: Maximum wallclock seconds
    
    Returns:
        dict: Results including factor (if found), windows tested, time, etc.
    """
    sqrt_N = gmpy2.isqrt(N)
    start_time = time.time()
    results = {
        "windows_tested": [],
        "factor_found": False,
        "factor": None,
        "total_candidates_tested": 0,
        "time_elapsed": 0
    }
    
    print(f"Starting adaptive window search")
    print(f"N = {N}")
    print(f"√N = {sqrt_N}")
    print(f"Max time: {max_time}s")
    
    for window_pct in WINDOW_SEQUENCE:
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            print(f"\nTimeout reached: {elapsed:.2f}s >= {max_time}s")
            break
            
        # ASYMMETRIC WINDOW: bias toward larger factor (q)
        # Based on observed enrichment pattern
        window_radius = int(sqrt_N * window_pct)
        # Ensure search_min stays positive
        search_min = max(gmpy2.mpz(2), sqrt_N - int(window_radius * WINDOW_BIAS_BELOW))
        search_max = sqrt_N + int(window_radius * WINDOW_BIAS_ABOVE)
        
        print(f"\n[Window {window_pct*100:.0f}%] Searching [{search_min}, {search_max}]")
        window_start = time.time()
        
        # Generate and score candidates
        print(f"  Generating {CANDIDATES_PER_WINDOW:,} candidates...")
        candidates = generate_qmc_candidates(search_min, search_max, CANDIDATES_PER_WINDOW)
        gen_time = time.time() - window_start
        print(f"  Generated in {gen_time:.2f}s")
        
        print(f"  Scoring with Z5D...")
        score_start = time.time()
        scored = score_candidates_z5d(candidates)
        score_time = time.time() - score_start
        print(f"  Scored {len(scored)} candidates in {score_time:.2f}s")
        
        # Test top-K via GCD
        top_k = max(1, int(len(scored) * TOP_K_FRACTION))
        print(f"  Testing top {top_k} candidates ({TOP_K_FRACTION*100:.1f}%) via GCD...")
        
        gcd_start = time.time()
        for i, (candidate, score) in enumerate(scored[:top_k]):
            results["total_candidates_tested"] += 1
            
            g = gmpy2.gcd(candidate, N)
            if 1 < g < N:
                # Factor found!
                results["factor_found"] = True
                results["factor"] = int(g)
                results["cofactor"] = int(N // g)
                results["window_pct"] = window_pct
                results["candidate_rank"] = i + 1
                results["candidate_score"] = float(score)
                results["time_elapsed"] = time.time() - start_time
                
                print(f"\n{'='*80}")
                print(f"FACTOR FOUND!")
                print(f"{'='*80}")
                print(f"Factor: {g}")
                print(f"Cofactor: {N // g}")
                print(f"Window: {window_pct*100:.0f}%")
                print(f"Rank: {i+1}/{top_k}")
                print(f"Score: {score:.4f}")
                print(f"Time: {results['time_elapsed']:.2f}s")
                print(f"{'='*80}")
                
                return results
        
        gcd_time = time.time() - gcd_start
        window_time = time.time() - window_start
        
        # Log window results
        window_result = {
            "window_pct": window_pct,
            "candidates_generated": len(candidates),
            "top_k_tested": top_k,
            "elapsed": time.time() - start_time,
            "window_time": window_time,
            "gen_time": gen_time,
            "score_time": score_time,
            "gcd_time": gcd_time
        }
        results["windows_tested"].append(window_result)
        
        current_elapsed = time.time() - start_time
        print(f"  Window completed in {window_time:.2f}s (total: {current_elapsed:.2f}s)")
    
    results["time_elapsed"] = time.time() - start_time
    print(f"\nSearch completed. No factor found.")
    print(f"Total time: {results['time_elapsed']:.2f}s")
    print(f"Total candidates tested: {results['total_candidates_tested']}")
    
    return results


def main():
    """
    Run the adaptive blind factorization experiment on test semiprimes.
    """
    print("="*80)
    print("ADAPTIVE BLIND FACTORIZATION EXPERIMENT")
    print("="*80)
    print()
    print("This experiment tests blind factorization using adaptive window")
    print("strategy with Z5D geometric resonance scoring.")
    print()
    
    # Output directory
    output_dir = Path(".")
    
    # Run on each test semiprime
    all_results = {}
    
    for name, test_case in TEST_SEMIPRIMES.items():
        print(f"\n{'='*80}")
        print(f"TEST CASE: {name}")
        print(f"Bits: {test_case['bits']}")
        print(f"{'='*80}")
        
        N = test_case["N"]
        p_true = test_case["p_true"]
        q_true = test_case["q_true"]
        
        # Run blind search (don't use p_true/q_true during search!)
        results = run_adaptive_window_search(N, MAX_WALLCLOCK_SECONDS)
        
        # Verify against ground truth (for validation only)
        if results["factor_found"]:
            factor = gmpy2.mpz(results["factor"])
            if factor == p_true or factor == q_true:
                print(f"\n✓ Correct factor found!")
                results["verification"] = "CORRECT"
            else:
                print(f"\n✗ Incorrect factor found!")
                results["verification"] = "INCORRECT"
        else:
            print(f"\n✗ No factor found")
            results["verification"] = "NOT_FOUND"
        
        # Store results
        all_results[name] = results
    
    # Save results to JSON
    output_file = output_dir / "adaptive_blind_results.json"
    
    def convert_mpz_to_str(obj):
        """Recursively convert gmpy2.mpz to strings for JSON serialization."""
        if isinstance(obj, gmpy2.mpz):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_mpz_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_mpz_to_str(item) for item in obj]
        else:
            return obj
    
    with open(output_file, 'w') as f:
        json_results = convert_mpz_to_str(all_results)
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\nSUMMARY:")
    for name, result in all_results.items():
        status = "✓ FOUND" if result["factor_found"] else "✗ NOT FOUND"
        time_str = f"{result['time_elapsed']:.2f}s"
        print(f"  {name}: {status} ({time_str})")
    
    return all_results


if __name__ == "__main__":
    main()
