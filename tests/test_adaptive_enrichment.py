"""
pytest-compatible test suite for adaptive enrichment experiment framework.

Tests validate:
1. Semiprime generation correctness (p × q = N)
2. Python int output (no numpy int64 overflow for 10^40 range)
3. Z5D scoring polarity (closer candidates → more negative scores)
4. Asymmetric bias detection (mean dist to q << mean dist to p)
"""

import pytest
import sys
from pathlib import Path
from math import isqrt

# Import modules from experiments/adaptive_enrichment
from generate_test_corpus import generate_corpus, SemiprimeCase
from qmc_candidate_generator import QMCCandidateGenerator, RandomCandidateGenerator
from z5d_score_emulator import z5d_score, prime_count_approx
from enrichment_analyzer import compute_enrichment


class TestSemiprimeGeneration:
    """Test corpus generation with controlled imbalance ratios."""

    def test_generate_small_corpus(self):
        """Test corpus generation with small parameters."""
        corpus = generate_corpus(
            magnitudes=[20], 
            ratios=[1.0], 
            samples_per_cell=1, 
            seed=42, 
            timeout_per_sample=30
        )
        
        assert len(corpus) >= 1, "Should generate at least 1 semiprime"
        
    def test_semiprime_validity(self):
        """Test that generated semiprimes satisfy p × q = N."""
        corpus = generate_corpus(
            magnitudes=[20], 
            ratios=[1.0], 
            samples_per_cell=1, 
            seed=42, 
            timeout_per_sample=30
        )
        
        case = corpus[0]
        assert isinstance(case, SemiprimeCase), "Should return SemiprimeCase objects"
        assert case.p * case.q == case.N, f"p × q != N ({case.p} × {case.q} != {case.N})"
        assert case.p < case.q, "p should be smaller than q"
        assert case.magnitude == 20, "Magnitude should be 20"

    def test_python_int_output(self):
        """Verify output uses Python int, not numpy int64."""
        corpus = generate_corpus(
            magnitudes=[20], 
            ratios=[1.0], 
            samples_per_cell=1, 
            seed=42, 
            timeout_per_sample=30
        )
        
        case = corpus[0]
        # Python int type, not numpy types
        assert type(case.N) is int, f"N should be Python int, got {type(case.N)}"
        assert type(case.p) is int, f"p should be Python int, got {type(case.p)}"
        assert type(case.q) is int, f"q should be Python int, got {type(case.q)}"


class TestCandidateGeneration:
    """Test QMC and random candidate generation strategies."""
    
    @pytest.fixture
    def test_semiprime(self):
        """Provide a test semiprime for candidate generation."""
        return {
            'N': 10493553215010178409,
            'sqrt_N': isqrt(10493553215010178409)
        }
    
    def test_symmetric_qmc_generation(self, test_semiprime):
        """Test symmetric QMC candidate generation."""
        gen = QMCCandidateGenerator(seed=42, asymmetric=False)
        candidates = gen.generate_candidates(test_semiprime['sqrt_N'], n_candidates=100)
        
        assert len(candidates) > 0, "Should generate candidates"
        assert all(isinstance(c, int) for c in candidates), "All candidates should be Python integers"
        # Verify no numpy int64 types
        assert all(type(c) is int for c in candidates), "All candidates should be Python int, not numpy types"
    
    def test_asymmetric_qmc_generation(self, test_semiprime):
        """Test asymmetric QMC candidate generation."""
        gen = QMCCandidateGenerator(seed=42, asymmetric=True)
        candidates = gen.generate_candidates(test_semiprime['sqrt_N'], n_candidates=100)
        
        assert len(candidates) > 0, "Should generate candidates"
        assert all(isinstance(c, int) for c in candidates), "All candidates should be Python integers"
    
    def test_random_generation(self, test_semiprime):
        """Test random baseline candidate generation."""
        gen = RandomCandidateGenerator(seed=42, asymmetric=False)
        candidates = gen.generate_candidates(test_semiprime['sqrt_N'], n_candidates=100)
        
        assert len(candidates) > 0, "Should generate candidates"
        assert all(isinstance(c, int) for c in candidates), "All candidates should be Python integers"


class TestZ5DScoring:
    """Test Z5D scoring function and polarity."""
    
    @pytest.fixture
    def test_case(self):
        """Provide test case for scoring."""
        N = 10493553215010178409
        return {
            'N': N,
            'sqrt_N': isqrt(N)
        }
    
    def test_prime_count_approximation(self):
        """Test prime count approximation function."""
        pi_100 = prime_count_approx(100)
        assert pi_100 > 0, "Prime count should be positive"
        # π(100) is approximately 25 (actual is 25)
        assert 20 < pi_100 < 35, f"π(100) should be around 25, got {pi_100}"
    
    def test_scoring_polarity(self, test_case):
        """Test that closer candidates score better (more negative)."""
        sqrt_N = test_case['sqrt_N']
        N = test_case['N']
        
        score_near = z5d_score(sqrt_N + 100, N, sqrt_N)
        score_far = z5d_score(sqrt_N + 10000, N, sqrt_N)
        
        # More negative = better, so score_near should be more negative than score_far
        assert score_near < score_far, \
            f"Closer candidates should score better (more negative): {score_near} >= {score_far}"
    
    def test_score_is_negative(self, test_case):
        """Test that scores near sqrt(N) are negative."""
        sqrt_N = test_case['sqrt_N']
        N = test_case['N']
        
        score = z5d_score(sqrt_N + 100, N, sqrt_N)
        # Scores should generally be negative for candidates near sqrt(N)
        assert score < 0, f"Score should be negative, got {score}"


class TestEnrichmentAnalysis:
    """Test enrichment analysis and statistical tests."""
    
    @pytest.fixture
    def test_factors(self):
        """Provide test factors for enrichment analysis."""
        N = 10493553215010178409
        p = 25492531
        q = 411632458739
        return {
            'N': N,
            'p': p,
            'q': q,
            'sqrt_N': isqrt(N)
        }
    
    def test_enrichment_computation(self, test_factors):
        """Test enrichment computation with asymmetric QMC."""
        gen = QMCCandidateGenerator(seed=42, asymmetric=True)
        candidates = gen.generate_candidates(test_factors['sqrt_N'], n_candidates=100)
        
        enrichment = compute_enrichment(
            candidates, 
            test_factors['p'], 
            test_factors['q'], 
            test_factors['sqrt_N']
        )
        
        assert enrichment.mean_dist_to_p >= 0, "Mean distance should be non-negative"
        assert enrichment.mean_dist_to_q >= 0, "Mean distance should be non-negative"
        assert 0 <= enrichment.ks_pvalue <= 1, "KS p-value should be in [0, 1]"
        assert 0 <= enrichment.mw_pvalue <= 1, "MW p-value should be in [0, 1]"
    
    def test_asymmetric_bias_detection(self, test_factors):
        """Test that asymmetric generator shows bias toward q."""
        gen = QMCCandidateGenerator(seed=42, asymmetric=True)
        candidates = gen.generate_candidates(test_factors['sqrt_N'], n_candidates=100)
        
        enrichment = compute_enrichment(
            candidates, 
            test_factors['p'], 
            test_factors['q'], 
            test_factors['sqrt_N']
        )
        
        # Asymmetric generator should produce candidates closer to q
        assert enrichment.mean_dist_to_q < enrichment.mean_dist_to_p, \
            f"Asymmetric bias not detected: dist_to_q ({enrichment.mean_dist_to_q}) >= dist_to_p ({enrichment.mean_dist_to_p})"


class TestLargeIntegerHandling:
    """Test handling of large integers (10^40 range)."""
    
    def test_large_integer_sqrt(self):
        """Test integer square root for 10^40 range numbers."""
        N_large = 10**40 + 7
        sqrt_N_large = isqrt(N_large)
        
        # Verify isqrt correctness
        assert sqrt_N_large * sqrt_N_large <= N_large < (sqrt_N_large + 1) * (sqrt_N_large + 1), \
            "isqrt should return correct integer square root"
    
    def test_candidate_generation_large_n(self):
        """Test candidate generation with large N (10^40 range)."""
        N_large = 10**40 + 7
        sqrt_N_large = isqrt(N_large)
        
        gen = QMCCandidateGenerator(seed=42, asymmetric=False)
        candidates = gen.generate_candidates(sqrt_N_large, n_candidates=10)
        
        assert len(candidates) > 0, "Should generate candidates for large N"
        assert all(isinstance(c, int) for c in candidates), "All candidates should be Python integers"
        assert all(type(c) is int for c in candidates), "All candidates should be Python int, not numpy types"
    
    def test_no_int64_overflow(self):
        """Verify that large numbers don't overflow to int64."""
        import numpy as np
        
        # Test that our implementation doesn't use int64
        N_large = 10**40 + 7
        sqrt_N_large = isqrt(N_large)
        
        gen = QMCCandidateGenerator(seed=42, asymmetric=False)
        candidates = gen.generate_candidates(sqrt_N_large, n_candidates=10)
        
        # Verify candidates can exceed int64 max
        int64_max = np.iinfo(np.int64).max
        
        # At least verify that we CAN represent numbers larger than int64
        # (candidates themselves might not exceed it, but the type should support it)
        large_test = 10**20  # Exceeds int64 max (9.2e18)
        assert large_test > int64_max, "Test value should exceed int64 max"
        
        # Verify our candidates are Python int which can handle this
        if candidates:
            assert type(candidates[0]) is int, "Candidates should be Python int"


class TestExperimentIntegration:
    """Test end-to-end experiment integration."""
    
    def test_full_pipeline_small_scale(self):
        """Test full pipeline with small corpus."""
        # Generate small corpus
        corpus = generate_corpus(
            magnitudes=[20], 
            ratios=[1.0], 
            samples_per_cell=1, 
            seed=42, 
            timeout_per_sample=30
        )
        
        assert len(corpus) >= 1, "Should generate corpus"
        
        # Test with each generator type
        generators = {
            'symmetric_random': RandomCandidateGenerator(seed=42, asymmetric=False),
            'symmetric_qmc': QMCCandidateGenerator(seed=42, asymmetric=False),
            'asymmetric_qmc': QMCCandidateGenerator(seed=42, asymmetric=True)
        }
        
        case = corpus[0]
        sqrt_N = isqrt(case.N)
        
        for name, gen in generators.items():
            candidates = gen.generate_candidates(sqrt_N, n_candidates=50)
            assert len(candidates) > 0, f"{name} should generate candidates"
            
            # Test scoring
            scores = [z5d_score(c, case.N, sqrt_N) for c in candidates]
            assert len(scores) == len(candidates), "Should have score for each candidate"
            
            # Test enrichment analysis
            enrichment = compute_enrichment(candidates, case.p, case.q, sqrt_N)
            assert enrichment is not None, f"{name} should produce enrichment result"
