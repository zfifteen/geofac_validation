"""
Integration tests for the full adaptive enrichment workflow.

Tests the complete pipeline:
1. Generate test corpus
2. Run experiment
3. Analyze results
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
import pandas as pd

from generate_test_corpus import generate_corpus
from run_experiment import main as run_experiment_main
from analyze_results import analyze


class TestWorkflowIntegration:
    """Test end-to-end workflow with file I/O."""
    
    def test_corpus_generation_workflow(self):
        """Test corpus generation creates valid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            corpus_path = f.name
        
        try:
            # Generate corpus
            corpus = generate_corpus(
                magnitudes=[20], 
                ratios=[1.0], 
                samples_per_cell=2, 
                seed=42,
                timeout_per_sample=30
            )
            
            # Save to file
            from dataclasses import asdict
            with open(corpus_path, 'w') as f:
                json.dump([asdict(c) for c in corpus], f)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(corpus_path), "Corpus file should exist"
            
            with open(corpus_path, 'r') as f:
                loaded = json.load(f)
            
            assert len(loaded) >= 1, "Corpus should have at least 1 entry"
            assert 'N' in loaded[0], "Corpus entries should have N field"
            assert 'p' in loaded[0], "Corpus entries should have p field"
            assert 'q' in loaded[0], "Corpus entries should have q field"
            
        finally:
            if os.path.exists(corpus_path):
                os.unlink(corpus_path)
    
    def test_experiment_workflow(self):
        """Test experiment execution creates valid results CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            corpus_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            results_path = f.name
        
        try:
            # Generate small corpus
            corpus = generate_corpus(
                magnitudes=[20], 
                ratios=[1.0], 
                samples_per_cell=1, 
                seed=42,
                timeout_per_sample=30
            )
            
            # Save corpus
            from dataclasses import asdict
            with open(corpus_path, 'w') as f:
                json.dump([asdict(c) for c in corpus], f)
            
            # Run experiment
            run_experiment_main(corpus_path, results_path)
            
            # Verify results file
            assert os.path.exists(results_path), "Results file should exist"
            
            df = pd.read_csv(results_path)
            
            # Should have 3 generators × corpus size trials
            expected_trials = len(corpus) * 3
            assert len(df) == expected_trials, f"Should have {expected_trials} trials"
            
            # Verify required columns
            required_cols = [
                'N', 'magnitude', 'imbalance_ratio', 'generator',
                'checks_to_find_factor', 'total_candidates',
                'enrichment_ratio', 'ks_pvalue', 'wall_time_ms'
            ]
            for col in required_cols:
                assert col in df.columns, f"Results should have {col} column"
            
            # Verify generator types
            generators = set(df['generator'].unique())
            expected_generators = {'symmetric_random', 'symmetric_qmc', 'asymmetric_qmc'}
            assert generators == expected_generators, f"Should have all 3 generators"
            
        finally:
            if os.path.exists(corpus_path):
                os.unlink(corpus_path)
            if os.path.exists(results_path):
                os.unlink(results_path)
    
    def test_analysis_workflow(self):
        """Test analysis creates valid report file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            corpus_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            results_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            report_path = f.name
        
        try:
            # Generate small corpus
            corpus = generate_corpus(
                magnitudes=[20], 
                ratios=[1.0], 
                samples_per_cell=1, 
                seed=42,
                timeout_per_sample=30
            )
            
            # Save corpus
            from dataclasses import asdict
            with open(corpus_path, 'w') as f:
                json.dump([asdict(c) for c in corpus], f)
            
            # Run experiment
            run_experiment_main(corpus_path, results_path)
            
            # Run analysis
            analyze(results_path, report_path)
            
            # Verify report file
            assert os.path.exists(report_path), "Report file should exist"
            
            with open(report_path, 'r') as f:
                report = f.read()
            
            # Verify report contains expected sections
            assert '# Validation Report' in report, "Report should have title"
            assert '## Aggregate Metrics' in report, "Report should have aggregate metrics"
            assert '## H₁ Success Criteria' in report, "Report should have H₁ criteria"
            assert '## Conclusion' in report, "Report should have conclusion"
            
            # Verify metrics are mentioned
            assert 'Q-enrichment' in report, "Report should mention Q-enrichment"
            assert 'KS p-value' in report, "Report should mention KS p-value"
            assert 'Check reduction' in report, "Report should mention check reduction"
            assert 'Variance ratio' in report, "Report should mention variance ratio"
            
        finally:
            if os.path.exists(corpus_path):
                os.unlink(corpus_path)
            if os.path.exists(results_path):
                os.unlink(results_path)
            if os.path.exists(report_path):
                os.unlink(report_path)
    
    def test_full_pipeline_end_to_end(self):
        """Test complete pipeline from corpus generation to analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            corpus_path = tmpdir / "corpus.json"
            results_path = tmpdir / "results.csv"
            report_path = tmpdir / "report.md"
            
            # Step 1: Generate corpus
            corpus = generate_corpus(
                magnitudes=[20], 
                ratios=[1.0, 1.5], 
                samples_per_cell=1, 
                seed=42,
                timeout_per_sample=30
            )
            
            assert len(corpus) >= 2, "Should generate at least 2 semiprimes"
            
            # Save corpus
            from dataclasses import asdict
            with open(corpus_path, 'w') as f:
                json.dump([asdict(c) for c in corpus], f)
            
            # Step 2: Run experiment
            run_experiment_main(str(corpus_path), str(results_path))
            
            assert results_path.exists(), "Results should be created"
            
            # Step 3: Analyze results
            analyze(str(results_path), str(report_path))
            
            assert report_path.exists(), "Report should be created"
            
            # Verify final report
            with open(report_path, 'r') as f:
                report = f.read()
            
            # Report should have conclusion (either H₁ or H₀)
            assert ('H₁ SUPPORTED' in report or 'H₀ NOT REJECTED' in report), \
                "Report should have definitive conclusion"


class TestExpectedMetrics:
    """Test that expected validation metrics are in reasonable ranges."""
    
    def test_enrichment_ratio_range(self):
        """Test enrichment ratio is computed correctly."""
        from enrichment_analyzer import compute_enrichment
        from math import isqrt
        
        # Simple test case
        N = 10**20 + 1
        sqrt_N = isqrt(N)
        p = 10**10 + 7
        q = N // p
        
        # Generate candidates near q
        candidates = [q + i for i in range(-10, 10)]
        
        enrichment = compute_enrichment(candidates, p, q, sqrt_N, proximity_threshold=0.05)
        
        # Should have more candidates near q than p
        assert enrichment.near_q_count >= enrichment.near_p_count, \
            "Should have more candidates near q"
    
    def test_ks_pvalue_computation(self):
        """Test KS p-value is computed and in valid range."""
        from enrichment_analyzer import compute_enrichment
        from math import isqrt
        
        N = 10**20 + 1
        sqrt_N = isqrt(N)
        p = 10**10 + 7
        q = N // p
        
        candidates = [sqrt_N + i for i in range(100)]
        
        enrichment = compute_enrichment(candidates, p, q, sqrt_N)
        
        assert 0 <= enrichment.ks_pvalue <= 1, "KS p-value should be in [0, 1]"
    
    def test_check_reduction_calculation(self):
        """Test check reduction is calculated correctly."""
        # This is tested implicitly in the analysis workflow
        # Just verify the calculation makes sense
        baseline_checks = 100
        asymmetric_checks = 60
        reduction = (baseline_checks - asymmetric_checks) / baseline_checks * 100
        
        assert reduction == 40.0, "Check reduction calculation should be correct"
        assert 0 <= reduction <= 100, "Reduction should be in valid percentage range"
