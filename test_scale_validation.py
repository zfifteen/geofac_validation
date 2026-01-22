#!/usr/bin/env python3
"""
Scale Validation Testing: 1e5 through 1e18

This script performs systematic testing of the adaptive blind factorization
algorithm across different semiprime scales to understand its performance
characteristics and identify any scale-dependent issues.
"""

import gmpy2
import time
import json
from pathlib import Path
import sys

# Import the main module
sys.path.insert(0, '.')
import run_adaptive_blind_factor as rabf


def generate_test_semiprime(bits):
    """Generate a test semiprime of specified bit length."""
    # Generate two primes of approximately equal size
    p_bits = bits // 2
    q_bits = bits - p_bits
    
    # Generate random primes
    p = gmpy2.next_prime(gmpy2.mpz(2) ** (p_bits - 1) + gmpy2.mpz_random(gmpy2.random_state(), 2 ** (p_bits - 1)))
    q = gmpy2.next_prime(gmpy2.mpz(2) ** (q_bits - 1) + gmpy2.mpz_random(gmpy2.random_state(), 2 ** (q_bits - 1)))
    
    N = p * q
    
    return {
        'N': N,
        'p': p,
        'q': q,
        'bits': int(gmpy2.bit_length(N)),
        'p_bits': int(gmpy2.bit_length(p)),
        'q_bits': int(gmpy2.bit_length(q))
    }


def run_scale_test(scale_name, N, p_true, q_true, bits, candidates_per_window=10000, max_time=120):
    """Run a single scale test."""
    print(f"\n{'='*80}")
    print(f"Testing: {scale_name}")
    print(f"Bits: {bits}")
    print(f"N = {N}")
    print(f"{'='*80}")
    
    # Configure for reasonable test time
    original_candidates = rabf.CANDIDATES_PER_WINDOW
    original_sequence = rabf.WINDOW_SEQUENCE
    original_timeout = rabf.MAX_WALLCLOCK_SECONDS
    
    rabf.CANDIDATES_PER_WINDOW = candidates_per_window
    rabf.WINDOW_SEQUENCE = [0.13, 0.20, 0.30]  # Test first 3 windows
    rabf.MAX_WALLCLOCK_SECONDS = max_time
    
    start_time = time.time()
    result = rabf.run_adaptive_window_search(N, max_time)
    elapsed = time.time() - start_time
    
    # Restore original settings
    rabf.CANDIDATES_PER_WINDOW = original_candidates
    rabf.WINDOW_SEQUENCE = original_sequence
    rabf.MAX_WALLCLOCK_SECONDS = original_timeout
    
    # Verify result
    if result['factor_found']:
        factor = gmpy2.mpz(result['factor'])
        if factor == p_true or factor == q_true:
            verification = "CORRECT"
        else:
            verification = "INCORRECT"
    else:
        verification = "NOT_FOUND"
    
    return {
        'scale_name': scale_name,
        'bits': bits,
        'N': str(N),
        'p': str(p_true),
        'q': str(q_true),
        'factor_found': result['factor_found'],
        'factor': str(result.get('factor', '')),
        'verification': verification,
        'windows_tested': len(result['windows_tested']),
        'total_candidates_tested': result['total_candidates_tested'],
        'time_elapsed': result['time_elapsed'],
        'candidates_per_window': candidates_per_window,
        'window_details': result['windows_tested']
    }


def main():
    """Run scale validation tests from 1e5 through 1e18."""
    
    print("="*80)
    print("SCALE VALIDATION TESTING")
    print("="*80)
    print()
    print("Testing adaptive blind factorization across different scales")
    print("Range: approximately 1e5 through 1e18")
    print()
    
    # Define test scales based on bit lengths
    # 1e5 ≈ 2^17, 1e18 ≈ 2^60
    test_scales = [
        {'name': '1e5_scale', 'bits': 17, 'desc': '~1e5'},
        {'name': '1e7_scale', 'bits': 23, 'desc': '~1e7'},
        {'name': '1e9_scale', 'bits': 30, 'desc': '~1e9'},
        {'name': '1e11_scale', 'bits': 37, 'desc': '~1e11'},
        {'name': '1e13_scale', 'bits': 43, 'desc': '~1e13'},
        {'name': '1e15_scale', 'bits': 50, 'desc': '~1e15'},
        {'name': '1e18_scale', 'bits': 60, 'desc': '~1e18'},
    ]
    
    all_results = []
    
    for scale in test_scales:
        print(f"\n{'='*80}")
        print(f"Generating test semiprime for {scale['desc']} ({scale['bits']} bits)")
        print(f"{'='*80}")
        
        # Generate test semiprime
        test_case = generate_test_semiprime(scale['bits'])
        
        # Run test
        result = run_scale_test(
            scale['name'],
            test_case['N'],
            test_case['p'],
            test_case['q'],
            test_case['bits'],
            candidates_per_window=5000,  # Keep tests fast
            max_time=60  # 1 minute per test
        )
        
        all_results.append(result)
        
        # Print summary
        print(f"\nResult: {result['verification']}")
        print(f"Time: {result['time_elapsed']:.2f}s")
        print(f"Candidates tested: {result['total_candidates_tested']}")
    
    # Save all results
    output_file = Path("scale_validation_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SCALE VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    
    # Generate summary report
    generate_summary_report(all_results)
    
    return all_results


def generate_summary_report(results):
    """Generate a summary findings document."""
    
    report_file = Path("SCALE_VALIDATION_FINDINGS.md")
    
    with open(report_file, 'w') as f:
        f.write("# Scale Validation Testing: Findings Report\n\n")
        
        # Conclusions first
        f.write("## Conclusions\n\n")
        
        success_count = sum(1 for r in results if r['verification'] == 'CORRECT')
        total_count = len(results)
        
        f.write(f"**Overall Success Rate:** {success_count}/{total_count} ")
        f.write(f"({100*success_count/total_count:.1f}%)\n\n")
        
        f.write("### Key Findings:\n\n")
        
        # Analyze timing trends
        times = [(r['bits'], r['time_elapsed']) for r in results]
        f.write("1. **Performance Scaling:**\n")
        for bits, elapsed in times:
            f.write(f"   - {bits}-bit: {elapsed:.2f}s\n")
        f.write("\n")
        
        # Analyze success by scale
        f.write("2. **Success by Scale:**\n")
        for r in results:
            status = "✓ FOUND" if r['verification'] == 'CORRECT' else "✗ NOT FOUND"
            f.write(f"   - {r['scale_name']}: {status}\n")
        f.write("\n")
        
        # Analyze candidates tested
        f.write("3. **Candidate Testing Efficiency:**\n")
        for r in results:
            f.write(f"   - {r['scale_name']}: {r['total_candidates_tested']} candidates tested\n")
        f.write("\n")
        
        # Technical details
        f.write("## Technical Details\n\n")
        
        for r in results:
            f.write(f"### {r['scale_name']}\n\n")
            f.write(f"- **Bit length:** {r['bits']}\n")
            f.write(f"- **Result:** {r['verification']}\n")
            f.write(f"- **Factor found:** {r['factor_found']}\n")
            f.write(f"- **Time elapsed:** {r['time_elapsed']:.2f}s\n")
            f.write(f"- **Windows tested:** {r['windows_tested']}\n")
            f.write(f"- **Total candidates tested:** {r['total_candidates_tested']}\n")
            f.write(f"- **Candidates per window:** {r['candidates_per_window']}\n")
            
            if r['window_details']:
                f.write("\n**Window Performance:**\n\n")
                for w in r['window_details']:
                    f.write(f"- Window {w['window_pct']*100:.0f}%: ")
                    f.write(f"{w['candidates_generated']} generated, ")
                    f.write(f"{w['top_k_tested']} tested, ")
                    f.write(f"{w['window_time']:.2f}s\n")
            
            f.write("\n")
        
        f.write("## Methodology\n\n")
        f.write("- **Candidate generation:** QMC Sobol sequence with 106-bit precision\n")
        f.write("- **Scoring:** Z5D geometric resonance\n")
        f.write("- **Window sequence:** 13%, 20%, 30% of √N (asymmetric)\n")
        f.write("- **Candidates per window:** 5,000 (reduced for testing speed)\n")
        f.write("- **Top-K tested:** 1% via GCD\n")
        f.write("- **Timeout:** 60 seconds per test\n")
        f.write("\n")
        
        f.write("## Notes\n\n")
        f.write("This is a preliminary validation with reduced parameters for speed.\n")
        f.write("Production runs would use 500,000 candidates per window and longer timeouts.\n")
        f.write("The purpose is to identify scale-dependent issues and performance characteristics.\n")
    
    print(f"Summary report saved to: {report_file}")


if __name__ == "__main__":
    main()
