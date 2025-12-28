#!/usr/bin/env python3
"""
Baseline Euclidean/Riemannian Metric for GVA Search

This module implements the existing Z5D-based geometric resonance scoring
as the baseline metric. It computes distances based on deviation from
Prime Number Theorem predictions, which represents a Riemannian-style
geometric distance on the prime manifold.

This is a SELF-CONTAINED copy of the Z5D logic for use in this experiment.
"""

import math


def z5d_n_est(p: int) -> int:
    """
    Estimate prime index n such that p(n) ≈ p.
    Uses asymptotic PNT approximation: pi(x) ≈ x / ln(x) * (1 + 1/ln(x) + 2/ln(x)^2)
    
    This is the inverse of the nth prime function.
    
    Args:
        p: Integer candidate value
        
    Returns:
        Estimated prime index n
    """
    if p < 2:
        return 1
    
    # Use Python's built-in math functions for reasonable precision
    # This works fine for numbers up to ~10^300 (1000 bits)
    ln_p = math.log(p)
    
    if ln_p <= 0:
        return 1
    
    # Prime counting function approximation
    inv_ln_p = 1.0 / ln_p
    n_est = (p / ln_p) * (1 + inv_ln_p + 2 * inv_ln_p * inv_ln_p)
    
    return max(1, int(n_est))


def z5d_predict_nth_prime(n: int) -> int:
    """
    Predict the nth prime using PNT-based approximation.
    
    Formula: p(n) ≈ n * (ln(n) + ln(ln(n)) - 1 + (ln(ln(n)) - 2) / ln(n))
    
    Args:
        n: Prime index
        
    Returns:
        Predicted value of the nth prime
    """
    if n <= 0:
        return 2
    if n == 1:
        return 2
    if n == 2:
        return 3
    if n == 3:
        return 5
    if n == 4:
        return 7
    if n == 5:
        return 11
    
    ln_n = math.log(n)
    if ln_n <= 0:
        return 2
    
    ln_ln_n = math.log(ln_n)
    
    # PNT approximation with correction terms
    correction = (ln_ln_n - 2) / ln_n
    predicted = n * (ln_n + ln_ln_n - 1 + correction)
    
    return max(2, int(predicted))


def compute_z5d_score(p: int, n_est: int = None) -> float:
    """
    Compute Z5D score as normalized log-relative deviation.
    
    Score = log10(|p - predicted_p| / p)
    
    Lower (more negative) scores indicate better alignment with Z5D model,
    meaning the candidate is closer to where a true prime should be.
    
    Args:
        p: Candidate value
        n_est: Optional pre-computed prime index estimate
        
    Returns:
        Z5D score (typically in range [-10, +10])
    """
    if p <= 0:
        return 0.0
    
    if n_est is None:
        n_est = z5d_n_est(p)
    
    predicted_p = z5d_predict_nth_prime(n_est)
    
    # Compute absolute difference
    diff = abs(p - predicted_p)
    
    if diff == 0:
        return -100.0  # Perfect prediction
    
    # Log-relative error: log10(|p - p'| / p)
    rel_error = diff / p
    
    if rel_error <= 0:
        return -100.0
    
    try:
        score = math.log10(rel_error)
        # Clamp to reasonable range
        return max(-100.0, min(100.0, score))
    except (ValueError, OverflowError):
        # Fallback for edge cases
        return 0.0


def euclidean_distance(a: int, b: int) -> int:
    """
    Simple Euclidean distance between two integers.
    
    Args:
        a, b: Integer values
        
    Returns:
        |a - b|
    """
    return abs(a - b)


def riemannian_distance(candidate: int, target: int) -> float:
    """
    Riemannian-style distance based on Z5D geometric manifold.
    
    This computes how far the candidate deviates from the expected
    position on the "prime manifold" defined by the PNT.
    
    Args:
        candidate: Candidate value to score
        target: Target value (typically sqrt(N) or known factor)
        
    Returns:
        Distance score (lower is better)
    """
    # Get Z5D scores for both
    score_candidate = compute_z5d_score(candidate)
    score_target = compute_z5d_score(target)
    
    # Distance is the difference in their geometric positions
    # We also incorporate the Euclidean distance
    euclidean = abs(candidate - target)
    geometric = abs(score_candidate - score_target)
    
    # Combine: geometric distance weighted by relative Euclidean distance
    # This creates a Riemannian-like metric where nearby points on the
    # manifold have similar Z5D scores
    if target > 0:
        normalized_euclidean = euclidean / target
    else:
        normalized_euclidean = 1.0
    
    # Final distance combines both geometric and Euclidean components
    return geometric + 0.1 * normalized_euclidean


def compute_gva_score(candidate: int, reference_point: int) -> float:
    """
    Compute GVA (Geometric Variational Analysis) score for a candidate.
    
    This is the main scoring function used in the baseline metric.
    Lower scores indicate better candidates for factorization.
    
    Args:
        candidate: Candidate factor value
        reference_point: Reference point (typically sqrt(N))
        
    Returns:
        GVA score (lower is better)
    """
    # Primary score: Z5D geometric resonance
    z5d = compute_z5d_score(candidate)
    
    # Secondary: Riemannian distance from reference
    riem_dist = riemannian_distance(candidate, reference_point)
    
    # Combined score (Z5D is primary, distance is secondary)
    # We want low Z5D scores (good geometric alignment)
    return z5d + 0.01 * riem_dist


if __name__ == "__main__":
    # Simple test
    test_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    print("Testing baseline metric on small primes:")
    print(f"{'p':<10} {'n_est':<10} {'predicted':<12} {'z5d_score':<12}")
    print("-" * 50)
    
    for p in test_values:
        n_est = z5d_n_est(p)
        predicted = z5d_predict_nth_prime(n_est)
        score = compute_z5d_score(p, n_est)
        print(f"{p:<10} {n_est:<10} {predicted:<12} {score:<12.4f}")
    
    # Test GVA scoring
    print("\nTesting GVA scoring around sqrt(143) = 11.96...")
    sqrt_143 = 12
    for candidate in [7, 11, 13, 17]:
        gva = compute_gva_score(candidate, sqrt_143)
        print(f"  Candidate {candidate}: GVA score = {gva:.4f}")
