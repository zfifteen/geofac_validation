#!/usr/bin/env python3
"""
Main Experiment Runner: Asymmetric Q-Factor Enrichment Falsification

Executes complete falsification experiment pipeline:
1. Generate stratified semiprime test set
2. Run baseline Monte Carlo enrichment trials
3. Run Z5D enrichment trials
4. Perform statistical analysis
5. Generate visualizations
6. Make falsification decision

Experiment ID: GEOFAC-ASYM-001
Version: 1.0
"""

import sys
import os
import yaml
import json
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate_test_set import generate_stratified_test_set, save_test_set, load_test_set
from baseline_mc_enrichment import run_baseline_enrichment_suite, save_baseline_results
from z5d_enrichment_test import run_z5d_enrichment_suite, save_z5d_results
from statistical_analysis import evaluate_falsification_criteria, save_statistical_analysis
from visualization import generate_all_visualizations


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_header(text: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def main():
    """Main experiment execution."""
    
    # Experiment metadata
    experiment_id = "GEOFAC-ASYM-001"
    version = "1.0"
    start_time = datetime.now()
    
    print_header(f"Asymmetric Q-Factor Enrichment Falsification - {experiment_id}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Version: {version}")
    
    # Setup paths
    base_dir = Path(__file__).parent
    config_dir = base_dir / "config"
    data_dir = base_dir / "data"
    results_dir = data_dir / "results"
    
    # Create directories if needed
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Load configurations
    print_header("Loading Configurations")
    semiprime_config = load_config(config_dir / "semiprime_generation.yaml")
    sampling_config = load_config(config_dir / "sampling_parameters.yaml")
    stats_config = load_config(config_dir / "statistical_thresholds.yaml")
    print("✓ Configurations loaded")
    
    # ========================================================================
    # PHASE 1: GENERATE SEMIPRIME TEST SET
    # ========================================================================
    
    print_header("Phase 1: Generate Semiprime Test Set")
    
    test_set_path = data_dir / "benchmark_semiprimes.json"
    
    if test_set_path.exists():
        print(f"Loading existing test set from {test_set_path}")
        # Load as JSON and extract semiprimes list
        with open(test_set_path, 'r') as f:
            data = json.load(f)
        semiprimes = data['semiprimes']
        print(f"✓ Loaded {len(semiprimes)} semiprimes")
    else:
        print("Generating new test set...")
        test_set = generate_stratified_test_set(semiprime_config)
        save_test_set(test_set, str(test_set_path))
        # Convert to list of dicts for processing
        semiprimes = [sp.to_dict() for sp in test_set]
        print(f"✓ Generated {len(semiprimes)} semiprimes")
    
    # ========================================================================
    # PHASE 2: BASELINE MONTE CARLO ENRICHMENT
    # ========================================================================
    
    print_header("Phase 2: Baseline Monte Carlo Enrichment")
    
    baseline_path = results_dir / "phase1_baseline_mc.json"
    
    if baseline_path.exists():
        print(f"Baseline results already exist at {baseline_path}")
        print("Skipping baseline phase (delete file to re-run)")
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        baseline_results = baseline_data['results']
    else:
        print("Running baseline enrichment trials...")
        print(f"  Trials per semiprime: {sampling_config['sample_sizes']['n_trials_per_semiprime']}")
        print(f"  Candidates per trial: {sampling_config['sample_sizes']['n_candidates']}")
        
        baseline_results_objs = run_baseline_enrichment_suite(
            semiprimes,
            n_trials=sampling_config['sample_sizes']['n_trials_per_semiprime'],
            n_candidates=sampling_config['sample_sizes']['n_candidates'],
            epsilon_pct=sampling_config['proximity_window']['epsilon_pct'],
            window_pct=sampling_config['search_window']['default_pct'],
            base_seed=sampling_config['random_seeds']['base_seed']
        )
        
        save_baseline_results(baseline_results_objs, str(baseline_path))
        baseline_results = [r.to_dict() for r in baseline_results_objs]
    
    print(f"✓ Baseline phase complete: {len(baseline_results)} measurements")
    
    # ========================================================================
    # PHASE 3: Z5D ENRICHMENT MEASUREMENT
    # ========================================================================
    
    print_header("Phase 3: Z5D Enrichment Measurement")
    
    z5d_path = results_dir / "phase2_z5d_enrichment.json"
    
    if z5d_path.exists():
        print(f"Z5D results already exist at {z5d_path}")
        print("Skipping Z5D phase (delete file to re-run)")
        with open(z5d_path, 'r') as f:
            z5d_data = json.load(f)
        z5d_results = z5d_data['results']
    else:
        print("Running Z5D enrichment trials...")
        print(f"  Trials per semiprime: {sampling_config['sample_sizes']['n_trials_per_semiprime']}")
        print(f"  Candidates per trial: {sampling_config['sample_sizes']['n_candidates']}")
        print(f"  Top percentage: {sampling_config['sample_sizes']['top_pct']}%")
        
        z5d_results_objs = run_z5d_enrichment_suite(
            semiprimes,
            n_trials=sampling_config['sample_sizes']['n_trials_per_semiprime'],
            n_candidates=sampling_config['sample_sizes']['n_candidates'],
            top_pct=sampling_config['sample_sizes']['top_pct'],
            epsilon_pct=sampling_config['proximity_window']['epsilon_pct'],
            window_pct=sampling_config['search_window']['default_pct'],
            base_seed=sampling_config['random_seeds']['base_seed']
        )
        
        save_z5d_results(z5d_results_objs, str(z5d_path))
        z5d_results = [r.to_dict() for r in z5d_results_objs]
    
    print(f"✓ Z5D phase complete: {len(z5d_results)} measurements")
    
    # ========================================================================
    # PHASE 4: STATISTICAL ANALYSIS & FALSIFICATION DECISION
    # ========================================================================
    
    print_header("Phase 4: Statistical Analysis & Falsification Decision")
    
    decision = evaluate_falsification_criteria(
        baseline_results,
        z5d_results,
        alpha=stats_config['significance']['alpha'],
        n_bootstrap=stats_config['statistical_tests']['bootstrap']['n_resamples']
    )
    
    # Save analysis
    analysis_path = results_dir / "falsification_decision.json"
    save_statistical_analysis(decision, str(analysis_path))
    
    # Print decision
    print("\n" + "=" * 80)
    print(f"  FALSIFICATION DECISION: {decision.decision}")
    print("=" * 80)
    print(f"\nConfidence: {decision.confidence * 100:.0f}%")
    print(f"\nPrimary Metrics:")
    print(f"  Q-enrichment: {decision.mean_q_enrichment:.2f}x "
          f"(95% CI: [{decision.q_enrichment_ci_lower:.2f}, {decision.q_enrichment_ci_upper:.2f}])")
    print(f"  P-enrichment: {decision.mean_p_enrichment:.2f}x "
          f"(95% CI: [{decision.p_enrichment_ci_lower:.2f}, {decision.p_enrichment_ci_upper:.2f}])")
    print(f"  Asymmetry ratio: {decision.mean_asymmetry_ratio:.2f}")
    
    print(f"\nStatistical Tests:")
    print(f"  Wilcoxon (q>5): p = {decision.wilcoxon_q_pvalue:.4f}")
    print(f"  Wilcoxon (p~1): p = {decision.wilcoxon_p_pvalue:.4f}")
    print(f"  Mann-Whitney U: p = {decision.mann_whitney_pvalue:.4f}, d = {decision.mann_whitney_effect_size:.2f}")
    
    print(f"\nFalsification Criteria:")
    print(f"  [{'✗' if decision.criterion_1_failed else '✓'}] Criterion 1: Q-enrichment > 2×")
    print(f"  [{'✗' if decision.criterion_2_failed else '✓'}] Criterion 2: P-enrichment < 3×")
    print(f"  [{'✗' if decision.criterion_3_failed else '✓'}] Criterion 3: Asymmetry ratio >= 2.0")
    print(f"  [{'✗' if decision.criterion_4_failed else '✓'}] Criterion 4: Replication across ranges")
    
    print(f"\nInterpretation:")
    print(f"  {decision.interpretation}")
    
    # ========================================================================
    # PHASE 5: GENERATE VISUALIZATIONS
    # ========================================================================
    
    print_header("Phase 5: Generate Visualizations")
    
    try:
        generate_all_visualizations(results_dir)
    except Exception as e:
        print(f"Warning: Visualization generation failed: {e}")
        print("Continuing without visualizations...")
    
    # ========================================================================
    # EXPERIMENT SUMMARY
    # ========================================================================
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("Experiment Complete")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"\nResults saved to: {results_dir}")
    print(f"  - Semiprime test set: {test_set_path}")
    print(f"  - Baseline results: {baseline_path}")
    print(f"  - Z5D results: {z5d_path}")
    print(f"  - Falsification decision: {analysis_path}")
    
    # Save experiment metadata
    metadata = {
        'experiment_id': experiment_id,
        'version': version,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'decision': decision.decision,
        'confidence': decision.confidence,
        'n_semiprimes': len(semiprimes),
        'n_baseline_measurements': len(baseline_results),
        'n_z5d_measurements': len(z5d_results)
    }
    
    metadata_path = results_dir / "experiment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  - Experiment metadata: {metadata_path}")
    
    return 0 if decision.decision != "FALSIFIED" else 1


if __name__ == '__main__':
    sys.exit(main())
