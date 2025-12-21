#!/usr/bin/env python3
"""
Quick validation test for experiment modules.

Tests basic functionality without running full experiment.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate_test_set import generate_balanced_semiprime, is_probable_prime
from baseline_mc_enrichment import generate_uniform_candidates, measure_enrichment
from z5d_enrichment_test import generate_qmc_candidates_sobol

import gmpy2
from gmpy2 import mpz


def test_primality_test():
    """Test Miller-Rabin primality testing."""
    print("\n1. Testing primality test...")
    
    # Known primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in primes:
        assert is_probable_prime(mpz(p)), f"Failed: {p} should be prime"
    
    # Known composites
    composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
    for n in composites:
        assert not is_probable_prime(mpz(n)), f"Failed: {n} should be composite"
    
    print("   ✓ Primality test working correctly")


def test_semiprime_generation():
    """Test semiprime generation with small bit lengths."""
    print("\n2. Testing semiprime generation...")
    
    # Generate 64-bit semiprime
    N, p, q = generate_balanced_semiprime(64, imbalance_pct=5.0, seed=42)
    
    # Verify N = p × q
    assert p * q == N, "Failed: N != p × q"
    
    # Verify bit length
    assert gmpy2.bit_length(N) == 64, f"Failed: Expected 64 bits, got {gmpy2.bit_length(N)}"
    
    # Verify both are prime
    assert is_probable_prime(p), "Failed: p is not prime"
    assert is_probable_prime(q), "Failed: q is not prime"
    
    # Verify p < q
    assert p < q, "Failed: p should be < q"
    
    print(f"   ✓ Generated 64-bit semiprime: N = {N}")
    print(f"   ✓ Factors: p = {p}, q = {q}")
    print(f"   ✓ Verification: p × q = N: {p * q == N}")


def test_uniform_candidate_generation():
    """Test uniform random candidate generation."""
    print("\n3. Testing uniform candidate generation...")
    
    search_min = mpz(1000)
    search_max = mpz(2000)
    n_samples = 100
    
    candidates = generate_uniform_candidates(search_min, search_max, n_samples, seed=42)
    
    # Verify count
    assert len(candidates) == n_samples, f"Failed: Expected {n_samples}, got {len(candidates)}"
    
    # Verify range
    for c in candidates:
        assert search_min <= c <= search_max, f"Failed: Candidate {c} out of range"
    
    # Verify all odd
    for c in candidates:
        assert c % 2 == 1, f"Failed: Candidate {c} is even"
    
    print(f"   ✓ Generated {n_samples} candidates in range [{search_min}, {search_max}]")
    print(f"   ✓ All candidates in range and odd")


def test_qmc_candidate_generation():
    """Test QMC (Sobol) candidate generation."""
    print("\n4. Testing QMC candidate generation...")
    
    search_min = mpz(1000)
    search_max = mpz(2000)
    n_samples = 100
    
    candidates = generate_qmc_candidates_sobol(search_min, search_max, n_samples, seed=42)
    
    # Verify count
    assert len(candidates) == n_samples, f"Failed: Expected {n_samples}, got {len(candidates)}"
    
    # Verify range
    for c in candidates:
        assert search_min <= c <= search_max, f"Failed: Candidate {c} out of range"
    
    # Verify all odd
    for c in candidates:
        assert c % 2 == 1, f"Failed: Candidate {c} is even"
    
    print(f"   ✓ Generated {n_samples} QMC candidates in range [{search_min}, {search_max}]")
    print(f"   ✓ All candidates in range and odd")


def test_enrichment_measurement():
    """Test enrichment measurement logic."""
    print("\n5. Testing enrichment measurement...")
    
    # Create synthetic candidates clustered near target
    target = mpz(5000)
    epsilon = mpz(50)
    
    # Generate candidates: 10 near target, 90 elsewhere
    candidates = []
    for i in range(10):
        candidates.append(target + mpz(i - 5))  # Near target
    for i in range(90):
        candidates.append(mpz(4000 + i * 10))  # Elsewhere
    
    # Baseline density (uniform in [4000, 5900])
    baseline_density = 100.0 / 1900.0  # 100 candidates in 1900 range
    
    # Measure enrichment
    count, enrichment = measure_enrichment(candidates, target, epsilon, baseline_density)
    
    print(f"   ✓ Candidates near target: {count}")
    print(f"   ✓ Expected (uniform): {baseline_density * 2 * float(epsilon):.2f}")
    print(f"   ✓ Enrichment ratio: {enrichment:.2f}x")
    
    # Should be enriched (10 found vs ~5 expected)
    assert enrichment > 1.5, f"Failed: Expected enrichment > 1.5x, got {enrichment:.2f}x"


def test_arbitrary_precision():
    """Test that arithmetic handles large numbers correctly."""
    print("\n6. Testing arbitrary precision arithmetic...")
    
    # Generate 256-bit semiprime
    N, p, q = generate_balanced_semiprime(256, imbalance_pct=10.0, seed=42)
    
    # Verify exact multiplication
    assert p * q == N, "Failed: Arbitrary precision multiplication"
    
    # Verify bit lengths
    assert gmpy2.bit_length(N) == 256, f"Failed: Expected 256 bits, got {gmpy2.bit_length(N)}"
    
    # Test sqrt
    sqrt_N = gmpy2.isqrt(N)
    assert p < sqrt_N < q, "Failed: p < √N < q not satisfied"
    
    print(f"   ✓ 256-bit arithmetic working correctly")
    print(f"   ✓ N has {gmpy2.bit_length(N)} bits")
    print(f"   ✓ p < √N < q verified")


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("  EXPERIMENT MODULE VALIDATION")
    print("=" * 80)
    
    try:
        test_primality_test()
        test_semiprime_generation()
        test_uniform_candidate_generation()
        test_qmc_candidate_generation()
        test_enrichment_measurement()
        test_arbitrary_precision()
        
        print("\n" + "=" * 80)
        print("  ✓ ALL VALIDATION TESTS PASSED")
        print("=" * 80)
        print("\nExperiment modules are working correctly.")
        print("Ready to run full experiment: python3 run_experiment.py")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ VALIDATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
