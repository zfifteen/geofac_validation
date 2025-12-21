#!/usr/bin/env python3
"""
Visualization Module for Falsification Experiment Results

Generates publication-quality outputs:
1. Enrichment comparison (box plots for baseline vs Z5D enrichment)
2. Asymmetry distribution (histogram of E_q / E_p ratios)
3. Confidence interval forest plots
4. Enrichment by bit range (bar charts)
5. Summary report (text)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def load_results(results_dir: Path) -> Dict:
    """Load all experiment results."""
    with open(results_dir / "phase1_baseline_mc.json") as f:
        baseline = json.load(f)
    
    with open(results_dir / "phase2_z5d_enrichment.json") as f:
        z5d = json.load(f)
    
    with open(results_dir / "falsification_decision.json") as f:
        decision = json.load(f)
    
    return {
        'baseline': baseline['results'],
        'z5d': z5d['results'],
        'decision': decision['decision']
    }


def plot_enrichment_comparison(results: Dict, output_path: Path):
    """
    Create box plots comparing baseline vs Z5D enrichment for p and q.
    
    Shows distribution of enrichment ratios across all trials.
    """
    baseline_results = results['baseline']
    z5d_results = results['z5d']
    
    # Extract enrichment values
    baseline_p = [r['enrichment_p'] for r in baseline_results]
    baseline_q = [r['enrichment_q'] for r in baseline_results]
    z5d_p = [r['z5d_enrichment_p'] for r in z5d_results]
    z5d_q = [r['z5d_enrichment_q'] for r in z5d_results]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # P-factor enrichment
    ax1.boxplot([baseline_p, z5d_p], labels=['Baseline\n(Uniform Random)', 'Z5D\n(Top 10%)'])
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='No enrichment')
    ax1.axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Falsification threshold')
    ax1.set_ylabel('Enrichment Ratio (p-factor)', fontsize=12)
    ax1.set_title('P-Factor Enrichment (Smaller Factor)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-factor enrichment
    ax2.boxplot([baseline_q, z5d_q], labels=['Baseline\n(Uniform Random)', 'Z5D\n(Top 10%)'])
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='No enrichment')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Falsification threshold')
    ax2.axhline(y=5.0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Claimed minimum')
    ax2.set_ylabel('Enrichment Ratio (q-factor)', fontsize=12)
    ax2.set_title('Q-Factor Enrichment (Larger Factor)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'enrichment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved enrichment_comparison.png")


def plot_asymmetry_distribution(results: Dict, output_path: Path):
    """
    Plot distribution of asymmetry ratios (E_q / E_p).
    
    Shows if asymmetric enrichment pattern is consistent.
    """
    z5d_results = results['z5d']
    decision = results['decision']
    
    # Extract asymmetry ratios (filter out inf values)
    asymmetry_ratios = [r['asymmetry_ratio'] for r in z5d_results 
                       if r['asymmetry_ratio'] != float('inf')]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(asymmetry_ratios, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add vertical lines for thresholds
    ax.axvline(x=2.0, color='red', linestyle='--', linewidth=2, 
               label='Falsification threshold (2.0)')
    ax.axvline(x=5.0, color='green', linestyle='--', linewidth=2,
               label='Claimed minimum (5.0)')
    ax.axvline(x=decision['mean_asymmetry_ratio'], color='blue', linestyle='-', linewidth=2,
               label=f'Observed mean ({decision["mean_asymmetry_ratio"]:.2f})')
    
    ax.set_xlabel('Asymmetry Ratio (E_q / E_p)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Asymmetry Ratios', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'asymmetry_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved asymmetry_distribution.png")


def plot_confidence_intervals(results: Dict, output_path: Path):
    """
    Forest plot showing 95% bootstrap confidence intervals.
    
    Visualizes uncertainty in enrichment estimates.
    """
    decision = results['decision']
    
    # Data for forest plot
    metrics = ['Q-enrichment', 'P-enrichment']
    means = [decision['mean_q_enrichment'], decision['mean_p_enrichment']]
    ci_lower = [decision['q_enrichment_ci_lower'], decision['p_enrichment_ci_lower']]
    ci_upper = [decision['q_enrichment_ci_upper'], decision['p_enrichment_ci_upper']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot CIs
    for i, (metric, mean, lower, upper) in enumerate(zip(metrics, means, ci_lower, ci_upper)):
        ax.plot([lower, upper], [i, i], 'o-', linewidth=2, markersize=8, label=metric)
        ax.plot(mean, i, 'D', markersize=10, color='darkblue')
    
    # Add reference lines
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='No enrichment')
    ax.axvline(x=2.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Q threshold')
    ax.axvline(x=3.0, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='P threshold')
    ax.axvline(x=5.0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Claimed minimum')
    
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Enrichment Ratio', fontsize=12)
    ax.set_title('95% Bootstrap Confidence Intervals', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved confidence_intervals.png")


def plot_enrichment_by_bit_range(results: Dict, output_path: Path):
    """
    Plot enrichment ratios grouped by bit-length range.
    
    Tests scale-invariance claim (Criterion 4).
    """
    z5d_results = results['z5d']
    
    # Group by bit range
    ranges = {}
    for r in z5d_results:
        sp_name = r['semiprime_name']
        range_name = sp_name.split('_')[0]
        
        if range_name not in ranges:
            ranges[range_name] = {'q': [], 'p': []}
        
        ranges[range_name]['q'].append(r['z5d_enrichment_q'])
        ranges[range_name]['p'].append(r['z5d_enrichment_p'])
    
    # Create figure
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Q-enrichment by range
    range_names = list(ranges.keys())
    q_means = [np.mean(ranges[r]['q']) for r in range_names]
    q_stds = [np.std(ranges[r]['q']) for r in range_names]
    
    ax1.bar(range_names, q_means, yerr=q_stds, alpha=0.7, color='steelblue', capsize=5)
    ax1.axhline(y=5.0, color='green', linestyle='--', linewidth=2, label='Claimed minimum (5.0)')
    ax1.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Falsification threshold (2.0)')
    ax1.set_ylabel('Q-Enrichment Ratio', fontsize=12)
    ax1.set_title('Q-Factor Enrichment by Bit Range', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # P-enrichment by range
    p_means = [np.mean(ranges[r]['p']) for r in range_names]
    p_stds = [np.std(ranges[r]['p']) for r in range_names]
    
    ax2.bar(range_names, p_means, yerr=p_stds, alpha=0.7, color='coral', capsize=5)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='No enrichment (1.0)')
    ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='Falsification threshold (3.0)')
    ax2.set_ylabel('P-Enrichment Ratio', fontsize=12)
    ax2.set_title('P-Factor Enrichment by Bit Range', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'enrichment_by_bit_range.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved enrichment_by_bit_range.png")


def generate_summary_report(results: Dict, output_path: Path):
    """
    Generate text summary report with key statistics.
    """
    decision = results['decision']
    
    report = f"""
================================================================================
ASYMMETRIC Q-FACTOR ENRICHMENT FALSIFICATION EXPERIMENT
Experiment ID: GEOFAC-ASYM-001
================================================================================

DECISION: {decision['decision']}
CONFIDENCE: {decision['confidence'] * 100:.0f}%

PRIMARY METRICS:
  Q-enrichment: {decision['mean_q_enrichment']:.2f}x 
                (95% CI: [{decision['q_enrichment_ci_lower']:.2f}, {decision['q_enrichment_ci_upper']:.2f}])
  P-enrichment: {decision['mean_p_enrichment']:.2f}x 
                (95% CI: [{decision['p_enrichment_ci_lower']:.2f}, {decision['p_enrichment_ci_upper']:.2f}])
  Asymmetry ratio: {decision['mean_asymmetry_ratio']:.2f}

STATISTICAL TESTS:
  Wilcoxon (q>5): p = {decision['wilcoxon_q_pvalue']:.4f}
  Wilcoxon (p~1): p = {decision['wilcoxon_p_pvalue']:.4f}
  Mann-Whitney U: p = {decision['mann_whitney_pvalue']:.4f}, d = {decision['mann_whitney_effect_size']:.2f}

FALSIFICATION CRITERIA:
  [{'✗' if decision['criterion_1_failed'] else '✓'}] Criterion 1: Q-enrichment > 2×
  [{'✗' if decision['criterion_2_failed'] else '✓'}] Criterion 2: P-enrichment < 3×
  [{'✗' if decision['criterion_3_failed'] else '✓'}] Criterion 3: Asymmetry ratio >= 2.0
  [{'✗' if decision['criterion_4_failed'] else '✓'}] Criterion 4: Replication across ranges
  
  Criteria failed: {decision['criteria_failed_count']}/4

INTERPRETATION:
  {decision['interpretation']}

================================================================================
"""
    
    with open(output_path / 'summary_report.txt', 'w') as f:
        f.write(report)
    
    print(f"  ✓ Saved summary_report.txt")


def generate_all_visualizations(results_dir: Path):
    """Generate all visualizations from experiment results."""
    print("\nGenerating visualizations...")
    
    # Create output directory
    viz_dir = results_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Load results
    results = load_results(results_dir)
    
    # Generate plots
    plot_enrichment_comparison(results, viz_dir)
    plot_asymmetry_distribution(results, viz_dir)
    plot_confidence_intervals(results, viz_dir)
    plot_enrichment_by_bit_range(results, viz_dir)
    generate_summary_report(results, viz_dir)
    
    print(f"\n✓ All visualizations saved to {viz_dir}")


if __name__ == '__main__':
    from pathlib import Path
    
    # Default path
    results_dir = Path(__file__).parent.parent / 'data' / 'results'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the experiment first: python3 run_experiment.py")
    else:
        generate_all_visualizations(results_dir)
