#!/usr/bin/env python3
"""
Test script for Gradient Descent Zoom Algorithm

This script validates the gradient descent implementation on known semiprimes
and verifies that it maintains arbitrary precision throughout.
"""

import gmpy2
import json
from pathlib import Path
from gradient_zoom import gradient_zoom


# Test semiprimes with known factors
TEST_SEMIPRIMES = {
    "small_balanced": {
        "N": gmpy2.mpz(87713),  # 239 × 367
        "p": gmpy2.mpz(239),
        "q": gmpy2.mpz(367),
        "bits": 17,
        "description": "Small balanced semiprime (17-bit)"
    },
    "medium_balanced": {
        "N": gmpy2.mpz(272019049),  # 16493 × 16493
        "p": gmpy2.mpz(16493),
        "q": gmpy2.mpz(16493),
        "bits": 29,
        "description": "Medium balanced semiprime (29-bit, perfect square)"
    },
    "large_imbalanced": {
        "N": gmpy2.mpz("2487311"),  # 1621 × 1535
        "p": gmpy2.mpz(1535),
        "q": gmpy2.mpz(1621),
        "bits": 22,
        "description": "Large imbalanced semiprime (22-bit)"
    },
    "N_127": {
        "N": gmpy2.mpz("137524771864208156028430259349934309717"),
        "p": gmpy2.mpz("10508623501177419659"),
        "q": gmpy2.mpz("13086849276577416863"),
        "bits": 127,
        "description": "Production semiprime (127-bit)"
    }
}


def verify_factor(N, factor):
    """Verify that factor is a valid factor of N."""
    if factor is None:
        return False, "No factor found"
    
    # Check that factor divides N
    if N % factor != 0:
        return False, f"Factor {factor} does not divide N"
    
    # Check that factor is not trivial
    if factor == 1 or factor == N:
        return False, f"Factor {factor} is trivial"
    
    cofactor = N // factor
    
    # Verify factorization
    if factor * cofactor != N:
        return False, f"Factorization incorrect: {factor} × {cofactor} ≠ {N}"
    
    return True, f"Valid factorization: {factor} × {cofactor} = {N}"


def run_test(name, test_data, **kwargs):
    """Run gradient zoom test on a single semiprime."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")
    print(f"Description: {test_data['description']}")
    print(f"N = {test_data['N']}")
    print(f"True factors: {test_data['p']} × {test_data['q']}")
    print(f"Bit length: {test_data['bits']}")
    print(f"{'='*80}\n")
    
    # Run gradient zoom
    result = gradient_zoom(test_data["N"], verbose=True, **kwargs)
    
    # Verify result
    print(f"\n{'='*80}")
    print(f"VERIFICATION")
    print(f"{'='*80}")
    
    if result["factor_found"]:
        valid, message = verify_factor(test_data["N"], result["factor"])
        print(f"✓ Factor found: {result['factor']}")
        print(f"✓ Cofactor: {result['cofactor']}")
        print(f"Verification: {message}")
        
        # Check if we found the expected factors
        if result["factor"] in [test_data["p"], test_data["q"]]:
            print(f"✓ Found expected factor!")
            status = "SUCCESS"
        else:
            print(f"⚠ Found different factorization than expected")
            status = "SUCCESS_DIFFERENT"
    else:
        print(f"✗ Factor not found")
        print(f"Convergence reason: {result['convergence_reason']}")
        status = "FAILED"
    
    print(f"Iterations: {result['iterations']}")
    print(f"Candidates tested: {result['total_candidates_tested']:,}")
    print(f"Time elapsed: {result['time_elapsed']:.2f}s")
    print(f"{'='*80}\n")
    
    return {
        "name": name,
        "status": status,
        "factor_found": result["factor_found"],
        "factor": str(result["factor"]) if result["factor"] else None,
        "cofactor": str(result["cofactor"]) if result["cofactor"] else None,
        "expected_p": str(test_data["p"]),
        "expected_q": str(test_data["q"]),
        "iterations": result["iterations"],
        "candidates_tested": result["total_candidates_tested"],
        "time_elapsed": result["time_elapsed"],
        "convergence_reason": result["convergence_reason"],
        "window_history": result["window_history"]
    }


def main():
    """Run all tests and generate report."""
    print("=" * 80)
    print("GRADIENT DESCENT ZOOM ALGORITHM TEST SUITE")
    print("=" * 80)
    print()
    
    results = []
    
    # Test 1: Small semiprime with default parameters
    results.append(run_test(
        "small_balanced",
        TEST_SEMIPRIMES["small_balanced"],
        initial_window_pct=0.13,
        zoom_factor=10,  # Smaller zoom for small numbers
        candidates_per_iteration=10_000,
        max_iterations=10
    ))
    
    # Test 2: Medium semiprime with default parameters
    results.append(run_test(
        "medium_balanced",
        TEST_SEMIPRIMES["medium_balanced"],
        initial_window_pct=0.13,
        zoom_factor=50,
        candidates_per_iteration=50_000,
        max_iterations=10
    ))
    
    # Test 3: Large imbalanced semiprime
    results.append(run_test(
        "large_imbalanced",
        TEST_SEMIPRIMES["large_imbalanced"],
        initial_window_pct=0.13,
        zoom_factor=10,
        candidates_per_iteration=10_000,
        max_iterations=10
    ))
    
    # Test 4: N_127 (production semiprime) - reduced parameters for testing
    results.append(run_test(
        "N_127",
        TEST_SEMIPRIMES["N_127"],
        initial_window_pct=0.13,
        zoom_factor=100,
        candidates_per_iteration=50_000,  # Reduced from 100k for faster testing
        max_iterations=8,  # Reduced for testing
        convergence_threshold_bits=32
    ))
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r["factor_found"])
    total_count = len(results)
    
    print(f"\nSuccess Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"\nDetailed Results:")
    
    for r in results:
        status_symbol = "✓" if r["factor_found"] else "✗"
        print(f"  {status_symbol} {r['name']}: {r['status']}")
        print(f"      Iterations: {r['iterations']}, Candidates: {r['candidates_tested']:,}, Time: {r['time_elapsed']:.2f}s")
    
    # Save results to JSON
    output_file = Path("gradient_zoom_test_results.json")
    
    # Convert gmpy2.mpz to strings for JSON serialization
    def serialize_mpz(obj):
        if isinstance(obj, gmpy2.mpz):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: serialize_mpz(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_mpz(item) for item in obj]
        return obj
    
    serializable_results = serialize_mpz(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 80)
    
    return success_count == total_count


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
