#!/usr/bin/env python3
"""
Baseline Monte Carlo Enrichment Measurement

Measures baseline (random) candidate distribution near factors p and q
using uniform random sampling. Provides control comparison for Z5D enrichment.

CRITICAL: Uses arbitrary-precision arithmetic (gmpy2) exclusively.
NO int64/uint64 - prevents silent overflow for 426-bit semiprimes.
"""

from gmpy2 import mpz, isqrt
import numpy as np
import random
from typing import Tuple, List
from dataclasses import dataclass, asdict
import json


@dataclass
class EnrichmentResult:
    """Container for enrichment measurement results."""
    semiprime_name: str
    trial_number: int
    n_candidates: int
    epsilon_pct: float
    
    # Target positions
    p_true: str
    q_true: str
    sqrt_N: str
    
    # Baseline counts
    candidates_near_p: int
    candidates_near_q: int
    expected_uniform_p: float
    expected_uniform_q: float
    
    # Enrichment ratios
    enrichment_p: float
    enrichment_q: float
    
    # Metadata
    search_min: str
    search_max: str
    window_pct: float
    
    def to_dict(self) -> dict:
        return asdict(self)


def measure_enrichment(
    candidates: List[mpz],
    target: mpz,
    epsilon: mpz,
    baseline_density: float
) -> Tuple[int, float]:
    """
    Measure enrichment factor in proximity window around target.
    
    Args:
        candidates: List of candidate values (gmpy2.mpz)
        target: Target factor (p or q)
        epsilon: Proximity window radius (absolute value)
        baseline_density: Expected density under uniform distribution
    
    Returns:
        (count, enrichment) where:
            - count: Number of candidates within epsilon of target
            - enrichment: Ratio relative to uniform baseline (1.0 = no enrichment)
    """
    # Count candidates within epsilon of target
    # Using arbitrary precision arithmetic
    proximity_count = 0
    for c in candidates:
        if abs(c - target) < epsilon:
            proximity_count += 1
    
    # Expected count under uniform distribution
    # window_size = 2 * epsilon
    # expected = baseline_density * window_size
    expected_uniform = baseline_density * (2 * float(epsilon))
    
    # Enrichment ratio
    if expected_uniform > 0:
        enrichment = proximity_count / expected_uniform
    else:
        enrichment = 0.0
    
    return proximity_count, enrichment


def generate_uniform_candidates(
    search_min: mpz,
    search_max: mpz,
    n_samples: int,
    seed: int = None
) -> List[mpz]:
    """
    Generate uniformly distributed candidates in search range.
    
    Uses Python's random module with arbitrary precision conversion.
    
    Args:
        search_min: Minimum search value (inclusive)
        search_max: Maximum search value (inclusive)
        n_samples: Number of candidates to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of candidate values (gmpy2.mpz)
    """
    rng = random.Random(seed)
    candidates = []
    
    # Convert to integers for random.randrange
    min_int = int(search_min)
    max_int = int(search_max)
    
    for _ in range(n_samples):
        # Generate random integer in range
        candidate = rng.randrange(min_int, max_int + 1)
        
        # Make odd (all primes except 2 are odd)
        if candidate % 2 == 0:
            candidate += 1
            if candidate > max_int:
                candidate -= 2
        
        # Convert to gmpy2.mpz for arbitrary precision
        candidates.append(mpz(candidate))
    
    return candidates


def run_baseline_enrichment_trial(
    N: mpz,
    p_true: mpz,
    q_true: mpz,
    window_pct: float = 15.0,
    n_candidates: int = 100000,
    epsilon_pct: float = 1.0,
    trial_num: int = 0,
    seed: int = None,
    semiprime_name: str = "unknown"
) -> EnrichmentResult:
    """
    Run single baseline enrichment measurement trial.
    
    Generates uniform random candidates and measures concentration near
    true factors p and q.
    
    Args:
        N: Semiprime N = p × q
        p_true: True smaller factor
        q_true: True larger factor
        window_pct: Search window as percentage of √N (default 15%)
        n_candidates: Number of candidates to generate (default 100k)
        epsilon_pct: Proximity window as percentage of √N (default 1%)
        trial_num: Trial number for identification
        seed: Random seed
        semiprime_name: Name/identifier for semiprime
    
    Returns:
        EnrichmentResult with all measurements
    """
    # Calculate search window
    sqrt_N = isqrt(N)
    window_radius = int(sqrt_N * window_pct / 100)
    search_min = sqrt_N - window_radius
    search_max = sqrt_N + window_radius
    search_width = search_max - search_min
    
    # Calculate proximity window (epsilon)
    epsilon = int(sqrt_N * epsilon_pct / 100)
    
    # Generate uniform random candidates
    candidates = generate_uniform_candidates(
        search_min, search_max, n_candidates, seed
    )
    
    # Calculate baseline density (uniform distribution)
    baseline_density = float(n_candidates) / float(search_width)
    
    # Measure enrichment near p
    count_p, enrichment_p = measure_enrichment(
        candidates, p_true, epsilon, baseline_density
    )
    
    # Measure enrichment near q
    count_q, enrichment_q = measure_enrichment(
        candidates, q_true, epsilon, baseline_density
    )
    
    # Expected counts under uniform distribution
    expected_uniform_p = baseline_density * (2 * float(epsilon))
    expected_uniform_q = baseline_density * (2 * float(epsilon))
    
    # Package results
    result = EnrichmentResult(
        semiprime_name=semiprime_name,
        trial_number=trial_num,
        n_candidates=n_candidates,
        epsilon_pct=epsilon_pct,
        p_true=str(p_true),
        q_true=str(q_true),
        sqrt_N=str(sqrt_N),
        candidates_near_p=count_p,
        candidates_near_q=count_q,
        expected_uniform_p=expected_uniform_p,
        expected_uniform_q=expected_uniform_q,
        enrichment_p=enrichment_p,
        enrichment_q=enrichment_q,
        search_min=str(search_min),
        search_max=str(search_max),
        window_pct=window_pct
    )
    
    return result


def run_baseline_enrichment_suite(
    semiprimes: List[dict],
    n_trials: int = 10,
    n_candidates: int = 100000,
    epsilon_pct: float = 1.0,
    window_pct: float = 15.0,
    base_seed: int = 42
) -> List[EnrichmentResult]:
    """
    Run baseline enrichment measurements across semiprime test set.
    
    Performs multiple trials per semiprime to establish statistical distribution
    of baseline (random) enrichment.
    
    Args:
        semiprimes: List of semiprime dictionaries with N, p, q, name
        n_trials: Number of independent trials per semiprime
        n_candidates: Candidates per trial
        epsilon_pct: Proximity window percentage
        window_pct: Search window percentage
        base_seed: Base random seed (incremented per trial)
    
    Returns:
        List of EnrichmentResult objects
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
            
            result = run_baseline_enrichment_trial(
                N, p, q,
                window_pct=window_pct,
                n_candidates=n_candidates,
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
                      f"E_p={result.enrichment_p:.2f}x, "
                      f"E_q={result.enrichment_q:.2f}x")
        
        # Summary statistics for this semiprime
        sp_results = [r for r in results if r.semiprime_name == name]
        mean_ep = np.mean([r.enrichment_p for r in sp_results])
        mean_eq = np.mean([r.enrichment_q for r in sp_results])
        print(f"  Mean baseline: E_p={mean_ep:.2f}x, E_q={mean_eq:.2f}x")
    
    return results


def save_baseline_results(results: List[EnrichmentResult], filepath: str):
    """Save baseline enrichment results to JSON."""
    data = {
        'results': [r.to_dict() for r in results],
        'count': len(results),
        'metadata': {
            'measurement': 'baseline_monte_carlo',
            'version': '1.0',
            'description': 'Uniform random candidate enrichment near p and q'
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {len(results)} baseline results to {filepath}")


def load_baseline_results(filepath: str) -> List[EnrichmentResult]:
    """Load baseline enrichment results from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [EnrichmentResult(**r) for r in data['results']]


if __name__ == '__main__':
    # Example usage with a simple test semiprime
    from generate_test_set import generate_balanced_semiprime
    
    print("Baseline Monte Carlo Enrichment Test")
    print("=" * 80)
    
    # Generate a 128-bit test semiprime
    N, p, q = generate_balanced_semiprime(128, imbalance_pct=10.0, seed=42)
    
    print(f"\nTest semiprime (128-bit):")
    print(f"N = {N}")
    print(f"p = {p}")
    print(f"q = {q}")
    
    # Run 3 baseline trials
    print("\nRunning baseline enrichment trials...")
    results = []
    for trial in range(3):
        result = run_baseline_enrichment_trial(
            N, p, q,
            n_candidates=10000,  # Reduced for testing
            trial_num=trial,
            seed=42 + trial,
            semiprime_name="test_128bit"
        )
        results.append(result)
        print(f"  Trial {trial}: E_p={result.enrichment_p:.2f}x, "
              f"E_q={result.enrichment_q:.2f}x")
    
    # Summary
    mean_ep = np.mean([r.enrichment_p for r in results])
    mean_eq = np.mean([r.enrichment_q for r in results])
    print(f"\nMean baseline enrichment:")
    print(f"  E_p = {mean_ep:.2f}x (expected ~1.0x for uniform random)")
    print(f"  E_q = {mean_eq:.2f}x (expected ~1.0x for uniform random)")
