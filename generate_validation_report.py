#!/usr/bin/env python3
"""
Generate Prospective Validation Analysis Report

This script analyzes the validation results and generates a comprehensive
markdown report with findings, visualizations, and conclusions.

USAGE
=====
python3 generate_validation_report.py \
    --validation results/prospective_validation_results.json \
    --benchmark results/benchmark_results.json \
    --output PROSPECTIVE_VALIDATION_REPORT.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_report(validation_data: Dict[str, Any],
                   benchmark_data: Dict[str, Any] = None) -> str:
    """Generate markdown report from validation and benchmark data."""
    
    report = []
    
    # Header
    report.append("# Prospective Validation Report: Gradient Descent Zoom Algorithm")
    report.append("")
    report.append(f"**Date:** {validation_data['metadata']['timestamp']}")
    report.append(f"**Dataset:** {validation_data['metadata']['dataset']}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    summary = validation_data['summary']
    report.append("## Executive Summary")
    report.append("")
    report.append(f"The gradient descent zoom algorithm was validated on **{summary['total_semiprimes']} prospective semiprimes** "
                 f"ranging from 80 to 140 bits, with pre-registered N values and controlled factor offset distributions.")
    report.append("")
    report.append(f"**Overall Results:**")
    report.append(f"- Success rate: **{summary['success_rate']*100:.1f}%** ({summary['successes']}/{summary['total_semiprimes']})")
    report.append(f"- Total runtime: {summary['total_time']:.2f}s ({summary['total_time']/60:.1f} minutes)")
    report.append(f"- Average time per semiprime: {summary['average_time']:.2f}s")
    report.append(f"- Total candidates tested: {summary['total_candidates_tested']:,}")
    report.append(f"- Average iterations: {summary['average_iterations']:.1f}")
    report.append("")
    
    # Success criteria evaluation
    success_rate = summary['success_rate']
    if success_rate >= 0.75:
        status = "✅ **STRETCH GOAL MET**"
        evidence = "Excellent evidence of algorithm viability"
    elif success_rate >= 0.50:
        status = "✅ **TARGET MET**"
        evidence = "Strong evidence of algorithm viability"
    elif success_rate >= 0.15:
        status = "✅ **MINIMUM MET**"
        evidence = "Weak evidence of algorithm viability"
    else:
        status = "❌ **BELOW MINIMUM**"
        evidence = "Insufficient evidence"
    
    report.append(f"**Verdict:** {status}")
    report.append(f"- {evidence}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Results by Bit Range
    report.append("## Results by Bit Range")
    report.append("")
    report.append("| Bit Range | Total | Successes | Failures | Success Rate |")
    report.append("|-----------|-------|-----------|----------|--------------|")
    
    for range_name, stats in summary['by_bit_range'].items():
        report.append(f"| {range_name} | {stats['total']} | {stats['successes']} | "
                     f"{stats['failures']} | {stats['success_rate']*100:.1f}% |")
    
    report.append("")
    
    # Results by Offset Type
    report.append("## Results by Offset Type")
    report.append("")
    report.append("| Offset Type | Total | Successes | Failures | Success Rate |")
    report.append("|-------------|-------|-----------|----------|--------------|")
    
    for offset_type, stats in summary['by_offset_type'].items():
        report.append(f"| {offset_type} | {stats['total']} | {stats['successes']} | "
                     f"{stats['failures']} | {stats['success_rate']*100:.1f}% |")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Detailed Results
    report.append("## Detailed Results")
    report.append("")
    
    results = validation_data['results']
    
    # Successes
    successes = [r for r in results if r.get('success', False)]
    if successes:
        report.append("### Successful Factorizations")
        report.append("")
        report.append("| ID | Bits | Offset Type | Iterations | Candidates | Time (s) |")
        report.append("|----|------|-------------|------------|------------|----------|")
        
        for r in successes:
            report.append(f"| {r['id']} | {r['bits']} | {r['offset_type']} | "
                         f"{r['iterations']} | {r['total_candidates_tested']:,} | {r['time_elapsed']:.2f} |")
        
        report.append("")
    
    # Failures
    failures = [r for r in results if not r.get('success', False)]
    if failures:
        report.append("### Failed Factorizations")
        report.append("")
        report.append("| ID | Bits | Offset Type | Reason | Iterations | Time (s) |")
        report.append("|----|------|-------------|--------|------------|----------|")
        
        for r in failures:
            reason = r.get('convergence_reason', r.get('error', 'unknown'))
            report.append(f"| {r['id']} | {r['bits']} | {r['offset_type']} | "
                         f"{reason} | {r.get('iterations', 0)} | {r.get('time_elapsed', 0):.2f} |")
        
        report.append("")
    
    report.append("---")
    report.append("")
    
    # Benchmarking comparison
    if benchmark_data and benchmark_data.get('comparison'):
        comparison = benchmark_data['comparison']
        
        report.append("## Algorithm Comparison")
        report.append("")
        report.append("| Algorithm | Success Rate | Avg Time (s) | Total Time (s) |")
        report.append("|-----------|--------------|--------------|----------------|")
        
        for alg, stats in comparison['by_algorithm'].items():
            if stats['total'] > 0:
                report.append(f"| {alg} | {stats['success_rate']*100:.1f}% "
                             f"({stats['successes']}/{stats['total']}) | "
                             f"{stats['avg_time']:.2f} | {stats['total_time']:.2f} |")
        
        report.append("")
        report.append("---")
        report.append("")
    
    # Analysis and Conclusions
    report.append("## Analysis and Conclusions")
    report.append("")
    
    # Convergence analysis
    avg_iterations = summary['average_iterations']
    report.append(f"### Convergence Characteristics")
    report.append("")
    report.append(f"- **Average iterations:** {avg_iterations:.1f}")
    report.append(f"- **Expected iterations (theory):** ~5 for 10⁹× window reduction")
    
    if avg_iterations > 7:
        report.append(f"- **Observation:** Higher than expected, suggesting gradient may require "
                     f"more iterations or is encountering local maxima")
    elif avg_iterations < 3:
        report.append(f"- **Observation:** Lower than expected, suggesting strong gradient "
                     f"convergence or early factor discovery")
    else:
        report.append(f"- **Observation:** Within expected range, consistent with theory")
    
    report.append("")
    
    # Performance analysis
    report.append(f"### Performance Analysis")
    report.append("")
    report.append(f"- **Average time:** {summary['average_time']:.2f}s")
    report.append(f"- **Expected time (theory):** 3-5 minutes for 127-bit semiprimes")
    
    if summary['average_time'] < 180:
        report.append(f"- **Observation:** Significantly faster than expected, likely due to "
                     f"smaller semiprimes or early convergence")
    elif summary['average_time'] > 600:
        report.append(f"- **Observation:** Slower than expected, may indicate gradient "
                     f"divergence or challenging offset configurations")
    
    report.append("")
    
    # Offset type analysis
    if summary['by_offset_type']:
        report.append(f"### Offset Type Analysis")
        report.append("")
        
        balanced_rate = summary['by_offset_type'].get('balanced', {}).get('success_rate', 0)
        moderate_rate = summary['by_offset_type'].get('moderate', {}).get('success_rate', 0)
        extreme_rate = summary['by_offset_type'].get('extreme', {}).get('success_rate', 0)
        
        report.append(f"- **Balanced factors** (|p-q| < 0.05√N): {balanced_rate*100:.1f}% success")
        report.append(f"- **Moderate offset** (0.10√N < |p-q| < 0.30√N): {moderate_rate*100:.1f}% success")
        report.append(f"- **Extreme offset** (0.50√N < |p-q| < 1.00√N): {extreme_rate*100:.1f}% success")
        report.append("")
        
        if balanced_rate > extreme_rate + 0.2:
            report.append(f"**Finding:** Gradient zoom performs significantly better on balanced "
                         f"factors, suggesting Z5D gradient is stronger near √N")
        elif extreme_rate > balanced_rate + 0.2:
            report.append(f"**Finding:** Surprisingly better performance on extreme offsets, "
                         f"suggesting Z5D asymmetry may favor distant factors")
        else:
            report.append(f"**Finding:** Performance relatively consistent across offset types")
        
        report.append("")
    
    # Overall conclusion
    report.append("### Conclusion")
    report.append("")
    
    if success_rate >= 0.50:
        report.append(f"The gradient descent zoom algorithm demonstrates **strong operational "
                     f"viability** with a {success_rate*100:.1f}% success rate on prospective "
                     f"validation. The algorithm successfully transforms the Coverage Paradox "
                     f"problem into a tractable optimization task.")
        report.append("")
        report.append(f"**Recommendation:** Proceed with integration into production pipeline "
                     f"and continue optimization of convergence parameters.")
    elif success_rate >= 0.15:
        report.append(f"The gradient descent zoom algorithm shows **promising results** with a "
                     f"{success_rate*100:.1f}% success rate, meeting minimum validation criteria. "
                     f"Further investigation is needed to understand failure modes and improve "
                     f"convergence reliability.")
        report.append("")
        report.append(f"**Recommendation:** Analyze failure cases, refine gradient clustering "
                     f"method, and consider multi-start strategies for robustness.")
    else:
        report.append(f"The gradient descent zoom algorithm achieved a {success_rate*100:.1f}% "
                     f"success rate, below minimum validation criteria. The approach requires "
                     f"significant refinement before operational deployment.")
        report.append("")
        report.append(f"**Recommendation:** Investigate fundamental issues with Z5D gradient "
                     f"quality, consider alternative clustering methods, or explore hybrid "
                     f"approaches combining gradient guidance with exhaustive search.")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Future Work
    report.append("## Future Work")
    report.append("")
    report.append("1. **Coppersmith Integration:** Implement Stage 3 handoff when window < N^(1/4)")
    report.append("2. **Multi-start Strategy:** Test multiple cluster centers to avoid local maxima")
    report.append("3. **Adaptive Zoom Factor:** Dynamically adjust zoom factor based on gradient strength")
    report.append("4. **Extended Validation:** Test on larger semiprimes (256+ bits)")
    report.append("5. **Comparative Analysis:** Benchmark against ECM and GNFS with external libraries")
    report.append("")
    report.append("---")
    report.append("")
    
    # References
    report.append("## References")
    report.append("")
    report.append("1. **ISSUE_43.md:** Prospective Validation Protocol Specification")
    report.append("2. **MASTER_FINDINGS.md:** Coverage Paradox Analysis")
    report.append("3. **Independent Research Report** (January 22, 2026)")
    report.append("")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Generate prospective validation analysis report"
    )
    parser.add_argument(
        '--validation',
        default='results/prospective_validation_results.json',
        help='Path to validation results JSON'
    )
    parser.add_argument(
        '--benchmark',
        default='results/benchmark_results.json',
        help='Path to benchmark results JSON (optional)'
    )
    parser.add_argument(
        '--output',
        default='PROSPECTIVE_VALIDATION_REPORT.md',
        help='Output markdown file'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATING VALIDATION REPORT")
    print("="*80)
    print(f"Validation data: {args.validation}")
    print(f"Benchmark data: {args.benchmark}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Load data
    validation_data = load_json(args.validation)
    
    benchmark_data = None
    if Path(args.benchmark).exists():
        benchmark_data = load_json(args.benchmark)
        print("✓ Loaded benchmark data")
    else:
        print("⚠ Benchmark data not found, skipping comparison section")
    
    # Generate report
    report = generate_report(validation_data, benchmark_data)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
