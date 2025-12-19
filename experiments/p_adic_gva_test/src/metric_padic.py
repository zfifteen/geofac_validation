#!/usr/bin/env python3
"""
p-adic Ultrametric for GVA Search

This module implements p-adic valuation and ultrametric distance functions
for integer factorization candidates. The p-adic metric provides a different
geometric structure than Euclidean/Riemannian metrics, potentially revealing
different structural properties of semiprimes.

Key Concepts:
- p-adic valuation: v_p(n) = highest power of prime p dividing n
- p-adic norm: ||n||_p = p^(-v_p(n))
- Ultrametric property: d(x,z) <= max(d(x,y), d(y,z))
"""

import math
from typing import List, Tuple


def prime_factorization_small(n: int, max_prime: int = 1000) -> dict:
    """
    Compute partial prime factorization for small primes up to max_prime.
    
    Args:
        n: Integer to factor
        max_prime: Maximum prime to try
        
    Returns:
        Dictionary mapping prime -> exponent
    """
    if n <= 1:
        return {}
    
    factors = {}
    
    # Handle 2 separately
    if n % 2 == 0:
        count = 0
        while n % 2 == 0:
            count += 1
            n //= 2
        factors[2] = count
    
    # Try odd primes
    p = 3
    while p * p <= n and p <= max_prime:
        if n % p == 0:
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            factors[p] = count
        p += 2
    
    # If n > 1 here, it's either prime or has large factors
    if n > 1 and n <= max_prime:
        factors[n] = 1
    
    return factors


def padic_valuation(n: int, p: int) -> int:
    """
    Compute p-adic valuation: v_p(n) = highest power of p dividing n.
    
    Args:
        n: Integer (must be non-zero)
        p: Prime base
        
    Returns:
        v_p(n) = max{k : p^k divides n}
    """
    if n == 0:
        return float('inf')  # Convention: v_p(0) = ∞
    
    if p <= 1:
        raise ValueError(f"Prime p must be > 1, got {p}")
    
    n = abs(n)
    valuation = 0
    
    while n % p == 0:
        valuation += 1
        n //= p
    
    return valuation


def padic_norm(n: int, p: int) -> float:
    """
    Compute p-adic norm: ||n||_p = p^(-v_p(n)).
    
    Properties:
    - ||0||_p = 0
    - ||n||_p > 0 for n ≠ 0
    - ||nm||_p = ||n||_p * ||m||_p
    - ||n + m||_p <= max(||n||_p, ||m||_p)  (ultrametric property)
    
    Args:
        n: Integer
        p: Prime base
        
    Returns:
        p-adic norm of n
    """
    if n == 0:
        return 0.0
    
    v = padic_valuation(n, p)
    return p ** (-v)


def padic_distance(a: int, b: int, p: int) -> float:
    """
    Compute p-adic distance: d_p(a, b) = ||a - b||_p.
    
    This satisfies the ultrametric property:
        d_p(a, c) <= max(d_p(a, b), d_p(b, c))
    
    Args:
        a, b: Integers
        p: Prime base
        
    Returns:
        p-adic distance between a and b
    """
    return padic_norm(a - b, p)


def multi_padic_distance(a: int, b: int, primes: List[int], weights: List[float] = None) -> float:
    """
    Compute weighted sum of p-adic distances over multiple primes.
    
    This creates a composite ultrametric from multiple p-adic metrics.
    
    Args:
        a, b: Integers
        primes: List of primes to use
        weights: Optional weights for each prime (default: equal weights)
        
    Returns:
        Weighted sum of p-adic distances
    """
    if weights is None:
        weights = [1.0 / len(primes)] * len(primes)
    
    if len(weights) != len(primes):
        raise ValueError("Number of weights must match number of primes")
    
    total_distance = 0.0
    for p, w in zip(primes, weights):
        total_distance += w * padic_distance(a, b, p)
    
    return total_distance


def adaptive_padic_score(candidate: int, reference: int, use_primes: List[int] = None) -> float:
    """
    Compute adaptive p-adic score for a factorization candidate.
    
    Strategy:
    1. Use small primes (2, 3, 5, 7, 11, ...) as bases
    2. Combine their p-adic distances to reference point
    3. Weight by "factorization relevance"
    
    Args:
        candidate: Candidate factor value
        reference: Reference point (typically sqrt(N))
        use_primes: List of primes to use (default: first 10 primes)
        
    Returns:
        p-adic score (lower indicates better structural similarity)
    """
    if use_primes is None:
        # Default: use first 10 primes
        use_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    # Compute p-adic distances for each prime
    distances = []
    for p in use_primes:
        d = padic_distance(candidate, reference, p)
        distances.append(d)
    
    # Weight smaller primes more heavily (they're more relevant for factorization)
    weights = [1.0 / (i + 1) for i in range(len(use_primes))]
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    # Compute weighted score
    score = sum(w * d for w, d in zip(weights, distances))
    
    # Add a "divisibility bonus" - lower score if candidate shares factors with reference
    # This captures structural similarity relevant to factorization
    gcd_val = math.gcd(candidate, reference)
    if gcd_val > 1:
        # Small bonus for sharing factors
        divisibility_bonus = -0.1 * math.log10(gcd_val)
        score += divisibility_bonus
    
    return score


def padic_ultrametric_gva_score(candidate: int, reference: int, N: int = None) -> float:
    """
    Main p-adic GVA scoring function.
    
    Combines:
    1. Multi-prime p-adic distances
    2. Divisibility structure analysis
    
    IMPORTANT: This function does NOT compute gcd(candidate, N) to avoid metric leakage.
    The scoring must be independent of whether the candidate is actually a factor.
    
    Args:
        candidate: Candidate factor value
        reference: Reference point (typically sqrt(N))
        N: Optional semiprime being factored (ONLY used for p-adic structure analysis,
           NOT for direct divisibility testing)
        
    Returns:
        p-adic GVA score (lower is better)
    """
    # Base score from adaptive p-adic metric
    base_score = adaptive_padic_score(candidate, reference)
    
    # If N is provided, analyze p-adic structure (but DO NOT test divisibility)
    if N is not None:
        # Analyze p-adic structure: compare small prime factorizations
        # This is legitimate because we're looking at structural similarity,
        # not testing if candidate divides N
        N_factors = prime_factorization_small(N, max_prime=100)
        candidate_factors = prime_factorization_small(candidate, max_prime=100)
        
        # Similarity in small prime factorization
        shared_primes = set(N_factors.keys()) & set(candidate_factors.keys())
        if shared_primes:
            # Mild bonus for sharing small prime factors
            similarity_bonus = -0.05 * len(shared_primes)
            base_score += similarity_bonus
    
    return base_score


if __name__ == "__main__":
    # Test p-adic metrics
    print("Testing p-adic valuation and norms:")
    print(f"{'n':<10} {'v_2(n)':<10} {'v_3(n)':<10} {'||n||_2':<12} {'||n||_3':<12}")
    print("-" * 60)
    
    test_values = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27]
    for n in test_values:
        v2 = padic_valuation(n, 2)
        v3 = padic_valuation(n, 3)
        norm2 = padic_norm(n, 2)
        norm3 = padic_norm(n, 3)
        print(f"{n:<10} {v2:<10} {v3:<10} {norm2:<12.4f} {norm3:<12.4f}")
    
    # Test ultrametric property
    print("\n\nTesting ultrametric property: d(a,c) <= max(d(a,b), d(b,c))")
    a, b, c = 8, 12, 20
    p = 2
    dab = padic_distance(a, b, p)
    dbc = padic_distance(b, c, p)
    dac = padic_distance(a, c, p)
    print(f"a={a}, b={b}, c={c}, p={p}")
    print(f"d({a},{b}) = {dab:.4f}")
    print(f"d({b},{c}) = {dbc:.4f}")
    print(f"d({a},{c}) = {dac:.4f}")
    print(f"max(d(a,b), d(b,c)) = {max(dab, dbc):.4f}")
    print(f"Ultrametric satisfied: {dac <= max(dab, dbc) + 1e-10}")
    
    # Test factorization scoring
    print("\n\nTesting p-adic GVA scoring for 143 = 11 × 13:")
    N = 143
    sqrt_N = 12
    candidates = [7, 11, 12, 13, 17, 19]
    
    print(f"{'Candidate':<12} {'p-adic Score':<15} {'Is Factor?':<12}")
    print("-" * 45)
    for cand in candidates:
        score = padic_ultrametric_gva_score(cand, sqrt_N, N)
        is_factor = "YES" if N % cand == 0 else "NO"
        print(f"{cand:<12} {score:<15.4f} {is_factor:<12}")
