#!/usr/bin/env python3
"""
Benchmark Gradient Zoom Against Other Factorization Algorithms

This script compares the gradient zoom algorithm against:
- Trial Division (baseline)
- Pollard's Rho
- (Note: ECM and GNFS require external libraries not available in this environment)

USAGE
=====
python3 benchmark_algorithms.py --dataset data/prospective_semiprimes.json --output results/benchmark_results.json
"""

import argparse
import json
import time
import gmpy2
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def trial_division(N: gmpy2.mpz, max_time: float = 60.0) -> Tuple[Optional[gmpy2.mpz], float]:
    """
    Trial division factorization.
    
    Args:
        N: Semiprime to factor
        max_time: Maximum time in seconds
    
    Returns:
        tuple: (factor or None, elapsed_time)
    """
    start_time = time.time()
    
    # Check small primes first
    if N % 2 == 0:
        return gmpy2.mpz(2), time.time() - start_time
    
    # Try odd divisors up to sqrt(N) or timeout
    limit = gmpy2.isqrt(N)
    d = gmpy2.mpz(3)
    
    while d <= limit:
        if time.time() - start_time > max_time:
            return None, time.time() - start_time
        
        if N % d == 0:
            return d, time.time() - start_time
        
        d += 2
    
    return None, time.time() - start_time


def pollard_rho(N: gmpy2.mpz, max_iterations: int = 1000000, max_time: float = 300.0) -> Tuple[Optional[gmpy2.mpz], float]:
    """
    Pollard's Rho factorization algorithm.
    
    Args:
        N: Semiprime to factor
        max_iterations: Maximum iterations
        max_time: Maximum time in seconds
    
    Returns:
        tuple: (factor or None, elapsed_time)
    """
    start_time = time.time()
    
    # Handle even numbers
    if N % 2 == 0:
        return gmpy2.mpz(2), time.time() - start_time
    
    # Pollard's Rho with Brent's cycle detection
    # Using f(x) = (x^2 + c) mod N
    c = gmpy2.mpz(1)
    x = gmpy2.mpz(2)
    y = gmpy2.mpz(2)
    d = gmpy2.mpz(1)
    
    iteration = 0
    while d == 1 and iteration < max_iterations:
        if time.time() - start_time > max_time:
            return None, time.time() - start_time
        
        # Tortoise move
        x = (x * x + c) % N
        
        # Hare move (twice as fast)
        y = (y * y + c) % N
        y = (y * y + c) % N
        
        # Check for cycle
        d = gmpy2.gcd(abs(x - y), N)
        
        iteration += 1
    
    elapsed = time.time() - start_time
    
    if d > 1 and d < N:
        return d, elapsed
    else:
        return None, elapsed


def benchmark_single_semiprime(N: gmpy2.mpz, bits: int, semiprime_id: str,
                               algorithms: List[str]) -> Dict[str, Any]:
    """
    Benchmark all algorithms on a single semiprime.
    
    Args:
        N: Semiprime to factor
        bits: Bit length
        semiprime_id: Identifier
        algorithms: List of algorithms to test
    
    Returns:
        dict: Benchmark results for all algorithms
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {semiprime_id} ({bits} bits)")
    print(f"N = {str(N)[:50]}...")
    print(f"{'='*80}")
    
    results = {
        'id': semiprime_id,
        'N': str(N),
        'bits': bits,
        'algorithms': {}
    }
    
    # Configure timeouts based on bit size
    if bits < 50:
        trial_timeout = 60
        rho_timeout = 60
    elif bits < 80:
        trial_timeout = 300
        rho_timeout = 300
    else:
        trial_timeout = 600
        rho_timeout = 600
    
    # Trial Division
    if 'trial_division' in algorithms:
        print(f"\n  Testing Trial Division (timeout: {trial_timeout}s)...")
        factor, elapsed = trial_division(N, max_time=trial_timeout)
        
        results['algorithms']['trial_division'] = {
            'success': factor is not None,
            'factor': str(factor) if factor else None,
            'time': elapsed,
            'timeout': elapsed >= trial_timeout
        }
        
        if factor:
            print(f"    ✓ SUCCESS in {elapsed:.2f}s")
        else:
            print(f"    ✗ TIMEOUT at {elapsed:.2f}s")
    
    # Pollard's Rho
    if 'pollard_rho' in algorithms:
        print(f"\n  Testing Pollard's Rho (timeout: {rho_timeout}s)...")
        factor, elapsed = pollard_rho(N, max_time=rho_timeout)
        
        results['algorithms']['pollard_rho'] = {
            'success': factor is not None,
            'factor': str(factor) if factor else None,
            'time': elapsed,
            'timeout': elapsed >= rho_timeout
        }
        
        if factor:
            print(f"    ✓ SUCCESS in {elapsed:.2f}s")
        else:
            print(f"    ✗ TIMEOUT/FAILED at {elapsed:.2f}s")
    
    print(f"{'='*80}")
    
    return results


def generate_comparison_report(benchmark_results: List[Dict[str, Any]],
                              validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comparison report between algorithms."""
    
    algorithms = ['gradient_zoom', 'trial_division', 'pollard_rho']
    
    comparison = {
        'by_algorithm': {},
        'by_semiprime': []
    }
    
    # Initialize algorithm stats
    for alg in algorithms:
        comparison['by_algorithm'][alg] = {
            'total': 0,
            'successes': 0,
            'failures': 0,
            'timeouts': 0,
            'total_time': 0,
            'avg_time': 0,
            'success_rate': 0
        }
    
    # Collect gradient zoom results
    gz_results = {r['id']: r for r in validation_results}
    
    # Process benchmark results
    for bench in benchmark_results:
        semiprime_id = bench['id']
        semiprime_comparison = {
            'id': semiprime_id,
            'bits': bench['bits'],
            'results': {}
        }
        
        # Gradient zoom
        if semiprime_id in gz_results:
            gz = gz_results[semiprime_id]
            semiprime_comparison['results']['gradient_zoom'] = {
                'success': gz.get('success', False),
                'time': gz.get('time_elapsed', 0),
                'timeout': gz.get('timeout', False)
            }
            
            alg_stats = comparison['by_algorithm']['gradient_zoom']
            alg_stats['total'] += 1
            alg_stats['successes'] += 1 if gz.get('success', False) else 0
            alg_stats['failures'] += 0 if gz.get('success', False) else 1
            alg_stats['timeouts'] += 1 if gz.get('timeout', False) else 0
            alg_stats['total_time'] += gz.get('time_elapsed', 0)
        
        # Other algorithms
        for alg in ['trial_division', 'pollard_rho']:
            if alg in bench['algorithms']:
                result = bench['algorithms'][alg]
                semiprime_comparison['results'][alg] = result
                
                alg_stats = comparison['by_algorithm'][alg]
                alg_stats['total'] += 1
                alg_stats['successes'] += 1 if result['success'] else 0
                alg_stats['failures'] += 0 if result['success'] else 1
                alg_stats['timeouts'] += 1 if result.get('timeout', False) else 0
                alg_stats['total_time'] += result['time']
        
        comparison['by_semiprime'].append(semiprime_comparison)
    
    # Calculate averages
    for alg, stats in comparison['by_algorithm'].items():
        if stats['total'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['total']
            stats['success_rate'] = stats['successes'] / stats['total']
    
    return comparison


def print_comparison_report(comparison: Dict[str, Any]):
    """Print formatted comparison report."""
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON REPORT")
    print("="*80)
    
    print("\nOverall Performance:")
    print(f"{'Algorithm':<20} {'Success Rate':<15} {'Avg Time':<15} {'Total Time':<15}")
    print("-"*80)
    
    for alg, stats in comparison['by_algorithm'].items():
        if stats['total'] > 0:
            success_pct = stats['success_rate'] * 100
            print(f"{alg:<20} {success_pct:>6.1f}% ({stats['successes']}/{stats['total']})  "
                  f"{stats['avg_time']:>10.2f}s    {stats['total_time']:>10.2f}s")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark gradient zoom against other factorization algorithms"
    )
    parser.add_argument(
        '--dataset',
        default='data/prospective_semiprimes.json',
        help='Path to prospective semiprimes dataset'
    )
    parser.add_argument(
        '--validation-results',
        default='results/prospective_validation_results.json',
        help='Path to gradient zoom validation results'
    )
    parser.add_argument(
        '--output',
        default='results/benchmark_results.json',
        help='Output file for benchmark results'
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['trial_division', 'pollard_rho'],
        help='Algorithms to benchmark against (trial_division, pollard_rho)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of semiprimes to test'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ALGORITHM BENCHMARKING")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Load dataset
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    if args.limit:
        dataset = dataset[:args.limit]
    
    # Run benchmarks
    benchmark_results = []
    for i, semiprime_data in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] Benchmarking {semiprime_data['id']}...")
        
        N = gmpy2.mpz(semiprime_data['N'])
        result = benchmark_single_semiprime(
            N,
            semiprime_data['bits'],
            semiprime_data['id'],
            args.algorithms
        )
        benchmark_results.append(result)
    
    # Load gradient zoom results if available
    validation_results = []
    if Path(args.validation_results).exists():
        with open(args.validation_results, 'r') as f:
            validation_data = json.load(f)
            validation_results = validation_data.get('results', [])
    
    # Generate comparison report
    if validation_results:
        comparison = generate_comparison_report(benchmark_results, validation_results)
        print_comparison_report(comparison)
    else:
        print("\nNote: Gradient zoom results not found. Skipping comparison.")
        comparison = None
    
    # Save results
    output_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': args.dataset,
            'algorithms': args.algorithms
        },
        'benchmark_results': benchmark_results,
        'comparison': comparison
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
