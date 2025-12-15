#!/usr/bin/env python3
"""
Generate visualization plots for Geofac-Z5D cross-validation results.
Creates two plots:
1. Z5D Score vs Scale (showing asymptotic convergence)
2. Amplitude vs Z5D Score (correlation analysis)
"""

import json
import sys
from pathlib import Path
import numpy as np

# Try to import matplotlib, provide helpful error if not installed
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("Error: matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


def load_results(artifacts_dir: Path) -> dict:
    """Load all crosscheck results organized by scale."""
    results = {}

    # Map of scale to filename pattern
    scale_files = [
        (100, "crosscheck_scale_100.jsonl"),
        (500, "crosscheck_scale_500.jsonl"),
        (1000, "crosscheck_scale_1000.jsonl"),
        (1233, "crosscheck_scale_1233.jsonl"),
    ]

    for scale, filename in scale_files:
        filepath = artifacts_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue

        scores_p = []
        scores_q = []
        amplitudes = []

        with open(filepath) as f:
            for line in f:
                if "_metadata" in line:
                    continue
                try:
                    record = json.loads(line)
                    score_p = record.get("z5d_score_p")
                    score_q = record.get("z5d_score_q")
                    amplitude = record.get("amplitude")

                    # Filter valid scores (negative log-relative errors)
                    if score_p is not None and -20 < score_p < 0:
                        scores_p.append(score_p)
                    if score_q is not None and -20 < score_q < 0:
                        scores_q.append(score_q)
                    if amplitude is not None:
                        amplitudes.append(amplitude)
                except json.JSONDecodeError:
                    continue

        if scores_p:
            results[scale] = {
                "scores_p": scores_p,
                "scores_q": scores_q,
                "amplitudes": amplitudes,
                "mean_score": np.mean(scores_p + scores_q),
                "std_score": np.std(scores_p + scores_q),
            }
            print(f"Scale 10^{scale}: {len(scores_p)} samples, mean score = {results[scale]['mean_score']:.3f}")

    return results


def plot_score_vs_scale(results: dict, output_path: Path):
    """Plot Z5D score vs scale showing asymptotic convergence."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scales = sorted(results.keys())
    means = [results[s]["mean_score"] for s in scales]
    stds = [results[s]["std_score"] for s in scales]

    # Plot with error bars
    ax.errorbar(scales, means, yerr=stds, fmt='o-', capsize=5, capthick=2,
                markersize=10, linewidth=2, color='#2563eb', ecolor='#93c5fd',
                label='Mean Z5D Score')

    # Add individual points for transparency
    for scale in scales:
        all_scores = results[scale]["scores_p"] + results[scale]["scores_q"]
        ax.scatter([scale] * len(all_scores), all_scores, alpha=0.3, s=30, color='#2563eb')

    # Formatting
    ax.set_xlabel('Scale (10^x)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z5D Score (log₁₀ relative error)', fontsize=12, fontweight='bold')
    ax.set_title('Z5D Prediction Accuracy Improves with Scale\n(Lower = Better)', fontsize=14, fontweight='bold')

    # Custom x-axis labels
    ax.set_xticks(scales)
    ax.set_xticklabels([f'10^{s}' for s in scales])

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add annotation showing improvement
    if len(scales) >= 2:
        improvement = means[0] - means[-1]
        final_error_pct = 10**(means[-1]) * 100  # Convert log10 relative error to %
        ax.annotate(f'{improvement:.1f} orders of magnitude\nimprovement\n({final_error_pct:.2e}% error)',
                   xy=(scales[-1], means[-1]), xytext=(scales[-1]-200, means[-1]+0.5),
                   fontsize=10, ha='right',
                   arrowprops=dict(arrowstyle='->', color='gray'))

    # Legend
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_amplitude_vs_score(results: dict, output_path: Path):
    """Plot amplitude vs Z5D score to show correlation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all data points with scale as color
    colors = {'100': '#22c55e', '500': '#3b82f6', '1000': '#a855f7', '1233': '#ef4444'}

    for scale in sorted(results.keys()):
        scores = results[scale]["scores_p"]
        amplitudes = results[scale]["amplitudes"][:len(scores)]

        color = colors.get(str(scale), '#6b7280')
        ax.scatter(amplitudes, scores, s=80, alpha=0.7, label=f'10^{scale}',
                  color=color, edgecolors='white', linewidths=0.5)

    # Compute overall correlation
    all_amps = []
    all_scores = []
    for scale in results:
        all_amps.extend(results[scale]["amplitudes"][:len(results[scale]["scores_p"])])
        all_scores.extend(results[scale]["scores_p"])

    if len(all_amps) > 2:
        corr = np.corrcoef(all_amps, all_scores)[0, 1]
        ax.annotate(f'Correlation: R = {corr:.3f}', xy=(0.05, 0.95),
                   xycoords='axes fraction', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Formatting
    ax.set_xlabel('Resonance Amplitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z5D Score (log₁₀ relative error)', fontsize=12, fontweight='bold')
    ax.set_title('Amplitude vs Prediction Accuracy Across Scales', fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(title='Scale', loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    artifacts_dir = project_root / "artifacts"

    # Create docs directory for images
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_results(artifacts_dir)

    if not results:
        print("No valid results found!")
        sys.exit(1)

    print("\nGenerating plots...")
    plot_score_vs_scale(results, docs_dir / "score_vs_scale.png")
    plot_amplitude_vs_score(results, docs_dir / "amplitude_vs_score.png")

    print("\nDone! Plots saved to docs/")


if __name__ == "__main__":
    main()
