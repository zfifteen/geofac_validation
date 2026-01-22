#!/usr/bin/env python3
"""
Run Prospective Validation on 20 Test Semiprimes

This script executes the full prospective validation protocol as specified in Issue #43.
It runs gradient zoom on all 20 test semiprimes and collects detailed statistics.

USAGE
=====
python3 run_prospective_validation.py --output results/prospective_validation_results.json

For quick testing (reduced parameters):
python3 run_prospective_validation.py --quick --output results/quick_test.json
"""

import argparse
import json
import time
import gmpy2
from pathlib import Path
from typing import Dict, Any, List
from gradient_zoom import gradient_zoom


def load_prospective_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load the prospective semiprime dataset."""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_single_validation(semiprime_data: Dict[str, Any], 
                         quick_mode: bool = False) -> Dict[str, Any]:
    """
    Run gradient zoom validation on a single semiprime.
    
    Args:
        semiprime_data: Semiprime metadata (N, id, bits, etc.)
        quick_mode: If True, use reduced parameters for faster testing
    
    Returns:
        dict: Validation results including success/failure, time, iterations, etc.
    """
    semiprime_id = semiprime_data['id']
    N = gmpy2.mpz(semiprime_data['N'])
    bits = semiprime_data['bits']
    offset_type = semiprime_data['offset_type']
    
    print(f"\n{'='*80}")
    print(f"VALIDATING: {semiprime_id}")
    print(f"{'='*80}")
    print(f"N = {str(N)[:50]}... ({bits} bits)")
    print(f"Offset type: {offset_type}")
    print(f"{'='*80}\n")
    
    # Configure parameters based on mode and semiprime size
    if quick_mode:
        # Quick mode: reduced parameters for testing
        candidates_per_iteration = 10_000
        max_iterations = 5
        max_time = 300  # 5 minutes
    else:
        # Production mode: full parameters
        if bits < 100:
            candidates_per_iteration = 50_000
            max_iterations = 8
            max_time = 1800  # 30 minutes
        else:
            candidates_per_iteration = 100_000
            max_iterations = 10
            max_time = 3600  # 1 hour
    
    start_time = time.time()
    
    try:
        # Run gradient zoom
        result = gradient_zoom(
            N,
            initial_window_pct=0.13,
            zoom_factor=100,
            candidates_per_iteration=candidates_per_iteration,
            max_iterations=max_iterations,
            convergence_threshold_bits=32,
            verbose=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Check timeout
        if elapsed_time >= max_time:
            result['timeout'] = True
            result['max_time'] = max_time
        else:
            result['timeout'] = False
        
        # Add metadata
        validation_result = {
            'id': semiprime_id,
            'N': str(N),
            'bits': bits,
            'offset_type': offset_type,
            'target_bits': semiprime_data.get('target_bits', bits),
            'success': result['factor_found'],
            'factor': str(result['factor']) if result['factor'] else None,
            'cofactor': str(result['cofactor']) if result['cofactor'] else None,
            'iterations': result['iterations'],
            'total_candidates_tested': result['total_candidates_tested'],
            'time_elapsed': result['time_elapsed'],
            'convergence_reason': result['convergence_reason'],
            'timeout': result.get('timeout', False),
            'window_history': result.get('window_history', []),
            'parameters': {
                'candidates_per_iteration': candidates_per_iteration,
                'max_iterations': max_iterations,
                'max_time': max_time,
                'quick_mode': quick_mode
            }
        }
        
        # Print summary
        print(f"\n{'='*80}")
        if result['factor_found']:
            print(f"✓ SUCCESS: {semiprime_id}")
            print(f"  Factor: {str(result['factor'])[:30]}...")
            print(f"  Cofactor: {str(result['cofactor'])[:30]}...")
        else:
            print(f"✗ FAILED: {semiprime_id}")
            print(f"  Reason: {result['convergence_reason']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Candidates: {result['total_candidates_tested']:,}")
        print(f"  Time: {result['time_elapsed']:.2f}s")
        print(f"{'='*80}\n")
        
        return validation_result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ ERROR: {semiprime_id}")
        print(f"  Exception: {type(e).__name__}: {str(e)}")
        print(f"  Time: {elapsed_time:.2f}s\n")
        
        return {
            'id': semiprime_id,
            'N': str(N),
            'bits': bits,
            'offset_type': offset_type,
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}",
            'time_elapsed': elapsed_time,
            'parameters': {
                'candidates_per_iteration': candidates_per_iteration,
                'max_iterations': max_iterations,
                'quick_mode': quick_mode
            }
        }


def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from validation results."""
    total = len(results)
    successes = sum(1 for r in results if r.get('success', False))
    failures = total - successes
    
    # Group by bit range
    bit_ranges = {
        '80-100': [r for r in results if 80 <= r['bits'] <= 100],
        '120-140': [r for r in results if 120 <= r['bits'] <= 140]
    }
    
    # Group by offset type
    offset_types = {}
    for r in results:
        ot = r.get('offset_type', 'unknown')
        if ot not in offset_types:
            offset_types[ot] = []
        offset_types[ot].append(r)
    
    # Calculate statistics
    total_time = sum(r.get('time_elapsed', 0) for r in results)
    total_candidates = sum(r.get('total_candidates_tested', 0) for r in results)
    avg_time = total_time / total if total > 0 else 0
    avg_iterations = sum(r.get('iterations', 0) for r in results) / total if total > 0 else 0
    
    summary = {
        'total_semiprimes': total,
        'successes': successes,
        'failures': failures,
        'success_rate': successes / total if total > 0 else 0,
        'total_time': total_time,
        'average_time': avg_time,
        'total_candidates_tested': total_candidates,
        'average_iterations': avg_iterations,
        'by_bit_range': {},
        'by_offset_type': {}
    }
    
    # Statistics by bit range
    for range_name, range_results in bit_ranges.items():
        if range_results:
            range_successes = sum(1 for r in range_results if r.get('success', False))
            summary['by_bit_range'][range_name] = {
                'total': len(range_results),
                'successes': range_successes,
                'failures': len(range_results) - range_successes,
                'success_rate': range_successes / len(range_results)
            }
    
    # Statistics by offset type
    for offset_type, type_results in offset_types.items():
        if type_results:
            type_successes = sum(1 for r in type_results if r.get('success', False))
            summary['by_offset_type'][offset_type] = {
                'total': len(type_results),
                'successes': type_successes,
                'failures': len(type_results) - type_successes,
                'success_rate': type_successes / len(type_results)
            }
    
    return summary


def print_summary_report(summary: Dict[str, Any]):
    """Print formatted summary report."""
    print("\n" + "="*80)
    print("PROSPECTIVE VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\nOverall Results:")
    print(f"  Total semiprimes: {summary['total_semiprimes']}")
    print(f"  Successes: {summary['successes']}")
    print(f"  Failures: {summary['failures']}")
    print(f"  Success rate: {summary['success_rate']*100:.1f}%")
    
    print(f"\nPerformance:")
    print(f"  Total time: {summary['total_time']:.2f}s ({summary['total_time']/60:.1f} minutes)")
    print(f"  Average time: {summary['average_time']:.2f}s")
    print(f"  Total candidates tested: {summary['total_candidates_tested']:,}")
    print(f"  Average iterations: {summary['average_iterations']:.1f}")
    
    print(f"\nBy Bit Range:")
    for range_name, stats in summary['by_bit_range'].items():
        print(f"  {range_name} bits:")
        print(f"    Success rate: {stats['success_rate']*100:.1f}% ({stats['successes']}/{stats['total']})")
    
    print(f"\nBy Offset Type:")
    for offset_type, stats in summary['by_offset_type'].items():
        print(f"  {offset_type}:")
        print(f"    Success rate: {stats['success_rate']*100:.1f}% ({stats['successes']}/{stats['total']})")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run prospective validation on test semiprimes"
    )
    parser.add_argument(
        '--dataset',
        default='data/prospective_semiprimes.json',
        help='Path to prospective semiprimes dataset'
    )
    parser.add_argument(
        '--output',
        default='results/prospective_validation_results.json',
        help='Output file for validation results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: use reduced parameters for faster testing'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of semiprimes to test (for debugging)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PROSPECTIVE VALIDATION - GRADIENT ZOOM ALGORITHM")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Quick mode: {args.quick}")
    if args.limit:
        print(f"Limit: {args.limit} semiprimes")
    print("="*80)
    
    # Load dataset
    dataset = load_prospective_dataset(args.dataset)
    
    if args.limit:
        dataset = dataset[:args.limit]
    
    print(f"\nLoaded {len(dataset)} semiprimes for validation")
    
    # Run validation on each semiprime
    results = []
    for i, semiprime_data in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] Processing {semiprime_data['id']}...")
        
        result = run_single_validation(semiprime_data, quick_mode=args.quick)
        results.append(result)
    
    # Generate summary
    summary = generate_summary_report(results)
    
    # Print summary
    print_summary_report(summary)
    
    # Save results
    output_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': args.dataset,
            'quick_mode': args.quick,
            'limit': args.limit
        },
        'summary': summary,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    
    # Success criteria check
    print("\n" + "="*80)
    print("SUCCESS CRITERIA EVALUATION (Issue #43)")
    print("="*80)
    success_rate = summary['success_rate']
    
    if success_rate >= 0.75:
        print(f"✓ STRETCH GOAL MET: {success_rate*100:.1f}% ≥ 75% (15/20)")
        status = "EXCELLENT"
    elif success_rate >= 0.50:
        print(f"✓ TARGET MET: {success_rate*100:.1f}% ≥ 50% (10/20)")
        status = "STRONG EVIDENCE"
    elif success_rate >= 0.15:
        print(f"✓ MINIMUM MET: {success_rate*100:.1f}% ≥ 15% (3/20)")
        status = "WEAK EVIDENCE"
    else:
        print(f"✗ BELOW MINIMUM: {success_rate*100:.1f}% < 15%")
        status = "INSUFFICIENT EVIDENCE"
    
    print(f"Status: {status}")
    print("="*80)
    
    return 0 if success_rate >= 0.15 else 1


if __name__ == "__main__":
    exit(main())
