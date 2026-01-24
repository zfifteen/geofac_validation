#!/usr/bin/env python3
"""
Quick validation test for adaptive enrichment experiment modules.

Tests basic functionality without running full experiment.
"""

import sys
import os
from math import isqrt

from generate_test_corpus import generate_corpus, SemiprimeCase
from qmc_candidate_generator import QMCCandidateGenerator, RandomCandidateGenerator
from z5d_score_emulator import z5d_score, prime_count_approx
from enrichment_analyzer import compute_enrichment


def test_semiprime_generation():
    """Test corpus generation with small parameters."""
    print("\n1. Testing semiprime generation...")
    
    # Generate small corpus
    corpus = generate_corpus(magnitudes=[20], ratios=[1.0], samples_per_cell=1, seed=42, timeout_per_sample=30)
    
    assert len(corpus) >= 1, "Failed: Should generate at least 1 semiprime"
    
    case = corpus[0]
    assert isinstance(case, SemiprimeCase), "Failed: Should return SemiprimeCase objects"
    assert case.p * case.q == case.N, f"Failed: p × q != N ({case.p} × {case.q} != {case.N})"
    assert case.p < case.q, "Failed: p should be smaller than q"
    assert case.magnitude == 20, "Failed: Magnitude should be 20"
    
    print(f"   ✓ Generated semiprime N={case.N} (p={case.p}, q={case.q})")


def test_candidate_generation():
    """Test QMC and random candidate generation."""
    print("\n2. Testing candidate generation...")
    
    # Test with a known semiprime
    N = 10493553215010178409
    sqrt_N = isqrt(N)
    
    # Test symmetric QMC
    gen_qmc_sym = QMCCandidateGenerator(seed=42, asymmetric=False)
    candidates_sym = gen_qmc_sym.generate_candidates(sqrt_N, n_candidates=100)
    assert len(candidates_sym) > 0, "Failed: Should generate candidates"
    assert all(isinstance(c, int) for c in candidates_sym), "Failed: All candidates should be integers"
    print(f"   ✓ Symmetric QMC generated {len(candidates_sym)} candidates")
    
    # Test asymmetric QMC
    gen_qmc_asym = QMCCandidateGenerator(seed=42, asymmetric=True)
    candidates_asym = gen_qmc_asym.generate_candidates(sqrt_N, n_candidates=100)
    assert len(candidates_asym) > 0, "Failed: Should generate candidates"
    print(f"   ✓ Asymmetric QMC generated {len(candidates_asym)} candidates")
    
    # Test random generator
    gen_random = RandomCandidateGenerator(seed=42, asymmetric=False)
    candidates_random = gen_random.generate_candidates(sqrt_N, n_candidates=100)
    assert len(candidates_random) > 0, "Failed: Should generate candidates"
    print(f"   ✓ Random generator generated {len(candidates_random)} candidates")


def test_z5d_scoring():
    """Test Z5D scoring function."""
    print("\n3. Testing Z5D scoring...")
    
    N = 10493553215010178409
    sqrt_N = isqrt(N)
    
    # Test prime count approximation
    pi_100 = prime_count_approx(100)
    assert pi_100 > 0, "Failed: Prime count should be positive"
    print(f"   ✓ Prime count approximation π(100) ≈ {pi_100:.2f}")
    
    # Test scoring near sqrt(N)
    score_near = z5d_score(sqrt_N + 100, N, sqrt_N)
    score_far = z5d_score(sqrt_N + 10000, N, sqrt_N)
    
    # More negative = better, so score_near should be more negative than score_far
    assert score_near < score_far, f"Failed: Closer candidates should score better (more negative): {score_near} >= {score_far}"
    print(f"   ✓ Score near sqrt(N): {score_near:.6f}")
    print(f"   ✓ Score far from sqrt(N): {score_far:.6f}")
    print(f"   ✓ Scoring polarity correct (more negative = better)")


def test_enrichment_analysis():
    """Test enrichment analysis."""
    print("\n4. Testing enrichment analysis...")
    
    N = 10493553215010178409
    p = 25492531
    q = 411632458739
    sqrt_N = isqrt(N)
    
    # Generate test candidates
    gen = QMCCandidateGenerator(seed=42, asymmetric=True)
    candidates = gen.generate_candidates(sqrt_N, n_candidates=100)
    
    # Compute enrichment
    enrichment = compute_enrichment(candidates, p, q, sqrt_N)
    
    assert enrichment.mean_dist_to_p >= 0, "Failed: Mean distance should be non-negative"
    assert enrichment.mean_dist_to_q >= 0, "Failed: Mean distance should be non-negative"
    assert 0 <= enrichment.ks_pvalue <= 1, "Failed: KS p-value should be in [0, 1]"
    
    print(f"   ✓ Mean distance to p: {enrichment.mean_dist_to_p:.6f}")
    print(f"   ✓ Mean distance to q: {enrichment.mean_dist_to_q:.6f}")
    print(f"   ✓ KS p-value: {enrichment.ks_pvalue:.6e}")
    
    # For asymmetric generator, expect bias toward q
    if enrichment.mean_dist_to_q < enrichment.mean_dist_to_p:
        print(f"   ✓ Asymmetric bias detected (closer to q)")


def test_large_integer_handling():
    """Test handling of large integers (10^40 range)."""
    print("\n5. Testing large integer handling...")
    
    # Test with 40-digit number
    N_large = 10**40 + 7  # Large N in 10^40 range
    sqrt_N_large = isqrt(N_large)
    
    assert sqrt_N_large * sqrt_N_large <= N_large < (sqrt_N_large + 1) * (sqrt_N_large + 1), \
        "Failed: isqrt should return correct integer square root"
    
    # Test candidate generation with large N
    gen = QMCCandidateGenerator(seed=42, asymmetric=False)
    candidates = gen.generate_candidates(sqrt_N_large, n_candidates=10)
    
    assert len(candidates) > 0, "Failed: Should generate candidates for large N"
    assert all(isinstance(c, (int, type(sqrt_N_large))) for c in candidates), \
        "Failed: All candidates should be integers"
    
    print(f"   ✓ Large integer sqrt: sqrt({N_large}) = {sqrt_N_large}")
    print(f"   ✓ Generated {len(candidates)} candidates for 10^40 range")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Adaptive Enrichment Experiment - Module Validation Tests")
    print("=" * 60)
    
    try:
        test_semiprime_generation()
        test_candidate_generation()
        test_z5d_scoring()
        test_enrichment_analysis()
        test_large_integer_handling()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
