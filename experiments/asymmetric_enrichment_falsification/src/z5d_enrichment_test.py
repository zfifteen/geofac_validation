#!/usr/bin/env python3
"""
Z5D Enrichment Test

Measures candidate enrichment near factors p and q using Z5D geometric
resonance scoring with QMC sampling.

Tests the asymmetric enrichment hypothesis:
- q-factor (larger): 5-10× enrichment expected
- p-factor (smaller): ~1× enrichment expected (no signal)

CRITICAL: Uses arbitrary-precision arithmetic (gmpy2/mpmath) exclusively.
NO int64/uint64 - prevents silent overflow for 426-bit semiprimes.
"""

import sys
import os
import gmpy2
from gmpy2 import mpz, isqrt
import numpy as np
from scipy.stats import qmc
import mpmath
from typing import Tuple, Dict, List
from dataclasses import dataclass, asdict
import json

# Add repository root to path for z5d_adapter import
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, repo_root)

from z5d_adapter import z5d_n_est, compute_z5d_score


@dataclass
class Z5DEnrichmentResult:
    """Container for Z5D enrichment measurement results."""
    semiprime_name: str
    trial_number: int
    n_candidates: int
    top_pct: float
    epsilon_pct: float
    
    # Target positions
    p_true: str
    q_true: str
    sqrt_N: str
    
    # Z5D enrichment counts
    top_candidates_near_p: int
    top_candidates_near_q: int
    
    # Baseline for comparison
    baseline_density: float
    expected_uniform_p: float
    expected_uniform_q: float
    
    # Z5D enrichment ratios (vs uniform baseline)
    z5d_enrichment_p: float
    z5d_enrichment_q: float
    
    # Asymmetry ratio
    asymmetry_ratio: float  # (E_q / E_p)
    
    # Metadata
    search_min: str
    search_max: str
    window_pct: float
    qmc_method: str
    
    def to_dict(self) -> dict:
        return asdict(self)


def generate_qmc_candidates_sobol(
    search_min: mpz,
    search_max: mpz,
    n_samples: int,
    seed: int = 42
) -> List[mpz]:
    """
    Generate quasi-Monte Carlo candidates using Sobol sequence.
    
    Replicates methodology from adversarial_test_adaptive.py for consistency.
    Uses 2D Sobol sequence with 106-bit fixed-point precision mapping.
    
    Args:
        search_min: Minimum search value (inclusive)
        search_max: Maximum search value (inclusive)
        n_samples: Number of candidates to generate
        seed: Random seed for scrambling
    
    Returns:
        List of candidate values (gmpy2.mpz)
    """
    # Initialize Sobol sampler with 2D space
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    qmc_samples = sampler.random(n=n_samples)
    
    # Convert to int for range arithmetic
    search_min_int = int(search_min)
    search_max_int = int(search_max)
    search_range_int = search_max_int - search_min_int
    
    # 106-bit fixed-point precision
    scale = 1 << 53  # 2^53 for high bits
    denom_bits = 106  # Total precision
    
    candidates = []
    for row in qmc_samples:
        # Map [0,1]² to 106-bit fixed-point
        hi = min(int(row[0] * scale), scale - 1)
        lo = min(int(row[1] * scale), scale - 1)
        
        # Combine into 106-bit value
        x = (hi << 53) | lo
        
        # Map to search range
        offset = (x * (search_range_int + 1)) >> denom_bits
        candidate = search_min_int + offset
        
        # Make odd
        if candidate % 2 == 0:
            candidate += 1
            if candidate > search_max_int:
                candidate -= 2
        
        # Convert to gmpy2.mpz
        candidates.append(mpz(candidate))
    
    return candidates


def score_candidates_z5d(candidates: List[mpz], N: mpz) -> List[Tuple[mpz, float]]:
    """
    Score all candidates using Z5D geometric resonance metric.
    
    Args:
        candidates: List of candidate values
        N: Semiprime N = p × q
    
    Returns:
        List of (candidate, score) tuples
    """
    scored = []
    
    for c in candidates:
        try:
            # Estimate which prime number this candidate is
            n_est = z5d_n_est(str(c))
            
            # Score based on deviation from PNT prediction
            score = compute_z5d_score(str(c), n_est)
            
            scored.append((c, score))
        except Exception as e:
            # Skip candidates that fail scoring
            continue
    
    return scored


def extract_top_candidates(
    scored_candidates: List[Tuple[mpz, float]],
    top_pct: float = 10.0
) -> List[mpz]:
    """
    Extract top percentage of candidates by Z5D score.
    
    Lower scores indicate better alignment with PNT predictions,
    so we take the candidates with LOWEST scores.
    
    Args:
        scored_candidates: List of (candidate, score) tuples
        top_pct: Percentage of top candidates to extract (default 10%)
    
    Returns:
        List of top-scoring candidates
    """
    # Sort by score (ascending - lower is better)
    sorted_candidates = sorted(scored_candidates, key=lambda x: x[1])
    
    # Extract top percentage
    n_top = max(1, int(len(sorted_candidates) * top_pct / 100))
    top = sorted_candidates[:n_top]
    
    # Return just the candidates (not scores)
    return [c for c, _ in top]


def measure_z5d_enrichment(
    top_candidates: List[mpz],
    p_true: mpz,
    q_true: mpz,
    epsilon: mpz,
    baseline_density: float
) -> Tuple[int, int, float, float]:
    """
    Measure enrichment in top Z5D candidates near p and q.
    
    Args:
        top_candidates: Top-scoring candidates from Z5D
        p_true: True smaller factor
        q_true: True larger factor
        epsilon: Proximity window radius
        baseline_density: Expected density under uniform distribution
    
    Returns:
        (count_p, count_q, enrichment_p, enrichment_q)
    """
    # Count top candidates near p
    count_p = sum(1 for c in top_candidates if abs(c - p_true) < epsilon)
    
    # Count top candidates near q
    count_q = sum(1 for c in top_candidates if abs(c - q_true) < epsilon)
    
    # Expected counts under uniform distribution
    expected_uniform = baseline_density * (2 * float(epsilon))
    
    # Enrichment ratios
    enrichment_p = count_p / expected_uniform if expected_uniform > 0 else 0.0
    enrichment_q = count_q / expected_uniform if expected_uniform > 0 else 0.0
    
    return count_p, count_q, enrichment_p, enrichment_q


def run_z5d_enrichment_trial(
    N: mpz,
    p_true: mpz,
    q_true: mpz,
    window_pct: float = 15.0,
    n_candidates: int = 100000,
    top_pct: float = 10.0,
    epsilon_pct: float = 1.0,
    trial_num: int = 0,
    seed: int = None,
    semiprime_name: str = "unknown"
) -> Z5DEnrichmentResult:
    """
    Run single Z5D enrichment measurement trial.
    
    Generates QMC candidates, scores with Z5D, and measures enrichment
    in top-scoring candidates near true factors.
    
    Args:
        N: Semiprime N = p × q
        p_true: True smaller factor
        q_true: True larger factor
        window_pct: Search window as percentage of √N (default 15%)
        n_candidates: Number of candidates to generate (default 100k)
        top_pct: Percentage of top candidates to analyze (default 10%)
        epsilon_pct: Proximity window as percentage of √N (default 1%)
        trial_num: Trial number for identification
        seed: Random seed
        semiprime_name: Name/identifier for semiprime
    
    Returns:
        Z5DEnrichmentResult with all measurements
    """
    # Calculate search window
    sqrt_N = isqrt(N)
    window_radius = int(sqrt_N * window_pct / 100)
    search_min = sqrt_N - window_radius
    search_max = sqrt_N + window_radius
    search_width = search_max - search_min
    
    # Calculate proximity window (epsilon)
    epsilon = int(sqrt_N * epsilon_pct / 100)
    
    # Generate QMC candidates using Sobol sequence
    candidates = generate_qmc_candidates_sobol(
        search_min, search_max, n_candidates, seed
    )
    
    # Score all candidates with Z5D
    scored_candidates = score_candidates_z5d(candidates, N)
    
    # Extract top percentage by score
    top_candidates = extract_top_candidates(scored_candidates, top_pct)
    
    # Calculate baseline density (for enrichment comparison)
    baseline_density = float(len(top_candidates)) / float(search_width)
    
    # Measure enrichment near p and q
    count_p, count_q, enrichment_p, enrichment_q = measure_z5d_enrichment(
        top_candidates, p_true, q_true, epsilon, baseline_density
    )
    
    # Calculate asymmetry ratio
    if enrichment_p > 0:
        asymmetry_ratio = enrichment_q / enrichment_p
    else:
        asymmetry_ratio = float('inf') if enrichment_q > 0 else 0.0
    
    # Expected counts under uniform distribution
    expected_uniform_p = baseline_density * (2 * float(epsilon))
    expected_uniform_q = baseline_density * (2 * float(epsilon))
    
    # Package results
    result = Z5DEnrichmentResult(
        semiprime_name=semiprime_name,
        trial_number=trial_num,
        n_candidates=n_candidates,
        top_pct=top_pct,
        epsilon_pct=epsilon_pct,
        p_true=str(p_true),
        q_true=str(q_true),
        sqrt_N=str(sqrt_N),
        top_candidates_near_p=count_p,
        top_candidates_near_q=count_q,
        baseline_density=baseline_density,
        expected_uniform_p=expected_uniform_p,
        expected_uniform_q=expected_uniform_q,
        z5d_enrichment_p=enrichment_p,
        z5d_enrichment_q=enrichment_q,
        asymmetry_ratio=asymmetry_ratio,
        search_min=str(search_min),
        search_max=str(search_max),
        window_pct=window_pct,
        qmc_method="Sobol"
    )
    
    return result


def run_z5d_enrichment_suite(
    semiprimes: List[dict],
    n_trials: int = 10,
    n_candidates: int = 100000,
    top_pct: float = 10.0,
    epsilon_pct: float = 1.0,
    window_pct: float = 15.0,
    base_seed: int = 42
) -> List[Z5DEnrichmentResult]:
    """
    Run Z5D enrichment measurements across semiprime test set.
    
    Args:
        semiprimes: List of semiprime dictionaries
        n_trials: Number of independent trials per semiprime
        n_candidates: Candidates per trial
        top_pct: Percentage of top candidates to analyze
        epsilon_pct: Proximity window percentage
        window_pct: Search window percentage
        base_seed: Base random seed
    
    Returns:
        List of Z5DEnrichmentResult objects
    """
    results = []
    trial_counter = 0
    
    for sp in semiprimes:
        # Parse semiprime data
        N = mpz(sp['N'])
        p = mpz(sp['p'])
        q = mpz(sp['q'])
        name = sp['name']
        
        print(f"\n{name} ({sp['bit_length']} bits):")
        
        for trial in range(n_trials):
            seed = base_seed + trial_counter
            
            result = run_z5d_enrichment_trial(
                N, p, q,
                window_pct=window_pct,
                n_candidates=n_candidates,
                top_pct=top_pct,
                epsilon_pct=epsilon_pct,
                trial_num=trial,
                seed=seed,
                semiprime_name=name
            )
            
            results.append(result)
            trial_counter += 1
            
            # Progress indicator
            if (trial + 1) % 3 == 0:
                print(f"  Trial {trial+1}/{n_trials}: "
                      f"E_p={result.z5d_enrichment_p:.2f}x, "
                      f"E_q={result.z5d_enrichment_q:.2f}x, "
                      f"Asymm={result.asymmetry_ratio:.2f}")
        
        # Summary statistics
        sp_results = [r for r in results if r.semiprime_name == name]
        mean_ep = np.mean([r.z5d_enrichment_p for r in sp_results])
        mean_eq = np.mean([r.z5d_enrichment_q for r in sp_results])
        mean_asymm = np.mean([r.asymmetry_ratio for r in sp_results if r.asymmetry_ratio != float('inf')])
        
        print(f"  Mean Z5D: E_p={mean_ep:.2f}x, E_q={mean_eq:.2f}x, Asymm={mean_asymm:.2f}")
    
    return results


def save_z5d_results(results: List[Z5DEnrichmentResult], filepath: str):
    """Save Z5D enrichment results to JSON."""
    data = {
        'results': [r.to_dict() for r in results],
        'count': len(results),
        'metadata': {
            'measurement': 'z5d_enrichment',
            'version': '1.0',
            'description': 'Z5D geometric resonance enrichment near p and q',
            'qmc_method': 'Sobol sequence with 106-bit precision'
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {len(results)} Z5D results to {filepath}")


def load_z5d_results(filepath: str) -> List[Z5DEnrichmentResult]:
    """Load Z5D enrichment results from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [Z5DEnrichmentResult(**r) for r in data['results']]


if __name__ == '__main__':
    print("Z5D Enrichment Test")
    print("=" * 80)
    print("\nThis module requires z5d_adapter.py from the repository root.")
    print("Run from experiments/asymmetric_enrichment_falsification/src/")
    print("\nExample usage in notebook or main experiment runner.")
