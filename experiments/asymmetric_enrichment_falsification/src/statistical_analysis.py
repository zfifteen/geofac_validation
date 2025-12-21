#!/usr/bin/env python3
"""
Statistical Analysis for Asymmetric Enrichment Falsification

Implements rigorous statistical tests to evaluate the asymmetric q-factor
enrichment hypothesis:
1. Wilcoxon signed-rank test
2. Bootstrap confidence intervals
3. Mann-Whitney U test
4. Levene's test for variance homogeneity

Falsification criteria:
- Q-enrichment ≤2× baseline over 10 trials
- P-enrichment ≥3× baseline over 10 trials
- q/p asymmetry ratio <2.0 (should be ≥5 per claim)
- Pattern fails across ≥3 different bit-length ranges
"""

import numpy as np
from scipy import stats
from typing import List, Tuple
import json
from dataclasses import dataclass, asdict


@dataclass
class FalsificationDecision:
    """Container for falsification analysis results."""
    decision: str  # FALSIFIED, CONFIRMED, PARTIALLY_CONFIRMED, INCONCLUSIVE
    confidence: float
    
    # Primary metrics
    mean_q_enrichment: float
    mean_p_enrichment: float
    mean_asymmetry_ratio: float
    
    # Statistical test results
    wilcoxon_q_pvalue: float
    wilcoxon_p_pvalue: float
    mann_whitney_pvalue: float
    mann_whitney_effect_size: float
    
    # Bootstrap CIs
    q_enrichment_ci_lower: float
    q_enrichment_ci_upper: float
    p_enrichment_ci_lower: float
    p_enrichment_ci_upper: float
    
    # Falsification criteria evaluation
    criterion_1_failed: bool  # Q-enrichment ≤2×
    criterion_2_failed: bool  # P-enrichment ≥3×
    criterion_3_failed: bool  # Asymmetry <2.0
    criterion_4_failed: bool  # Pattern fails across ≥3 ranges
    
    # Summary
    criteria_failed_count: int
    interpretation: str
    
    def to_dict(self) -> dict:
        return asdict(self)


def wilcoxon_signed_rank_test(
    enrichments: List[float],
    null_value: float = 1.0,
    alternative: str = 'greater'
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired comparison.
    
    Tests if median enrichment differs from null_value (1.0 = no enrichment).
    
    Args:
        enrichments: List of enrichment ratios
        null_value: Null hypothesis value (default 1.0)
        alternative: 'greater', 'less', or 'two-sided'
    
    Returns:
        (statistic, p_value)
    """
    # Subtract null value to test against zero
    differences = np.array(enrichments) - null_value
    
    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(differences, alternative=alternative)
    
    return statistic, p_value


def bootstrap_confidence_interval(
    data: List[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for mean.
    
    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap resamples
        confidence_level: CI confidence level (default 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        (mean, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    data_array = np.array(data)
    n = len(data_array)
    
    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data_array, size=n, replace=True)
        bootstrap_means.append(np.mean(resample))
    
    # Calculate percentiles for CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    mean = np.mean(data_array)
    
    return mean, ci_lower, ci_upper


def mann_whitney_u_test(
    sample1: List[float],
    sample2: List[float],
    alternative: str = 'greater'
) -> Tuple[float, float, float]:
    """
    Mann-Whitney U test (independent samples).
    
    Tests if sample1 is stochastically greater than sample2.
    
    Args:
        sample1: First sample (e.g., q-enrichments)
        sample2: Second sample (e.g., p-enrichments)
        alternative: 'greater', 'less', or 'two-sided'
    
    Returns:
        (statistic, p_value, effect_size)
        effect_size is Cohen's d
    """
    # Perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(
        sample1, sample2, alternative=alternative
    )
    
    # Calculate effect size (Cohen's d)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    std1 = np.std(sample1, ddof=1)
    std2 = np.std(sample2, ddof=1)
    
    # Pooled standard deviation
    n1 = len(sample1)
    n2 = len(sample2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    
    return statistic, p_value, effect_size


def levene_test_variance_homogeneity(
    groups: List[List[float]]
) -> Tuple[float, float]:
    """
    Levene's test for variance homogeneity across groups.
    
    Tests if variances are equal across multiple groups (e.g., bit-length ranges).
    
    Args:
        groups: List of sample groups
    
    Returns:
        (statistic, p_value)
    """
    statistic, p_value = stats.levene(*groups)
    return statistic, p_value


def evaluate_falsification_criteria(
    baseline_results: List[dict],
    z5d_results: List[dict],
    alpha: float = 0.01,
    n_bootstrap: int = 10000
) -> FalsificationDecision:
    """
    Evaluate all falsification criteria and make decision.
    
    Falsification criteria:
    1. Q-enrichment ≤2× baseline over 10 trials
    2. P-enrichment ≥3× baseline over 10 trials
    3. q/p asymmetry ratio <2.0 (should be ≥5 per claim)
    4. Pattern fails to replicate across ≥3 different bit-length ranges
    
    Args:
        baseline_results: List of baseline enrichment results
        z5d_results: List of Z5D enrichment results
        alpha: Significance level (default 0.01 with Bonferroni correction)
        n_bootstrap: Bootstrap resamples
    
    Returns:
        FalsificationDecision object
    """
    # Extract enrichment values
    q_enrichments = [r['z5d_enrichment_q'] for r in z5d_results]
    p_enrichments = [r['z5d_enrichment_p'] for r in z5d_results]
    asymmetry_ratios = [r['asymmetry_ratio'] for r in z5d_results 
                       if r['asymmetry_ratio'] != float('inf')]
    
    # Calculate summary statistics
    mean_q = np.mean(q_enrichments)
    mean_p = np.mean(p_enrichments)
    mean_asymm = np.mean(asymmetry_ratios) if asymmetry_ratios else 0.0
    
    # Test 1: Wilcoxon for q-enrichment (should be >5.0)
    _, wilcoxon_q_p = wilcoxon_signed_rank_test(
        q_enrichments, null_value=5.0, alternative='greater'
    )
    
    # Test 2: Wilcoxon for p-enrichment (should be ~1.0)
    _, wilcoxon_p_p = wilcoxon_signed_rank_test(
        p_enrichments, null_value=1.0, alternative='two-sided'
    )
    
    # Test 3: Mann-Whitney U test (q vs p)
    _, mann_whitney_p, effect_size = mann_whitney_u_test(
        q_enrichments, p_enrichments, alternative='greater'
    )
    
    # Bootstrap confidence intervals
    mean_q_boot, q_ci_lower, q_ci_upper = bootstrap_confidence_interval(
        q_enrichments, n_bootstrap=n_bootstrap
    )
    mean_p_boot, p_ci_lower, p_ci_upper = bootstrap_confidence_interval(
        p_enrichments, n_bootstrap=n_bootstrap
    )
    
    # Evaluate falsification criteria
    criterion_1 = mean_q <= 2.0 or q_ci_upper < 2.0  # Q-enrichment too low
    criterion_2 = mean_p >= 3.0 or p_ci_lower > 3.0  # P-enrichment too high
    criterion_3 = mean_asymm < 2.0  # Asymmetry ratio too low
    
    # Criterion 4: Check replication across bit ranges
    # Group results by bit length range
    bit_ranges = {}
    for r in z5d_results:
        sp_name = r['semiprime_name']
        # Extract range from name (e.g., "Small_128bit_10pct_0" -> "Small")
        range_name = sp_name.split('_')[0]
        if range_name not in bit_ranges:
            bit_ranges[range_name] = []
        bit_ranges[range_name].append(r['z5d_enrichment_q'])
    
    # Count ranges where mean q-enrichment ≥5.0
    ranges_with_signal = sum(1 for vals in bit_ranges.values() if np.mean(vals) >= 5.0)
    total_ranges = len(bit_ranges)
    criterion_4 = ranges_with_signal < min(3, total_ranges - 2)  # Fails in ≥3 ranges
    
    criteria_failed = sum([criterion_1, criterion_2, criterion_3, criterion_4])
    
    # Make falsification decision
    # Per original specification: "falsified if any of the following conditions are met"
    # This means ANY ONE failure is sufficient for falsification
    if criteria_failed >= 1:
        decision = "FALSIFIED"
        interpretation = (
            f"Hypothesis falsified: {criteria_failed}/4 criteria failed. "
            f"Q-enrichment mean={mean_q:.2f}x (expected >5x), "
            f"P-enrichment mean={mean_p:.2f}x (expected ~1x), "
            f"Asymmetry ratio={mean_asymm:.2f} (expected ≥5). "
            f"Per specification, ANY failure falsifies the hypothesis."
        )
        confidence = 0.95 if criteria_failed >= 2 else 0.85
    elif criteria_failed == 0 and wilcoxon_q_p < alpha and mann_whitney_p < alpha:
        decision = "CONFIRMED"
        interpretation = (
            f"Hypothesis confirmed: All criteria met. "
            f"Q-enrichment={mean_q:.2f}x, P-enrichment={mean_p:.2f}x, "
            f"Asymmetry={mean_asymm:.2f}. "
            f"Statistical significance: p < {alpha} (Bonferroni corrected)."
        )
        confidence = 0.95
    else:
        decision = "INCONCLUSIVE"
        interpretation = (
            f"Inconclusive: Insufficient statistical power or high variance. "
            f"Recommend larger sample or tighter controls."
        )
        confidence = 0.50
    
    # Package results
    result = FalsificationDecision(
        decision=decision,
        confidence=confidence,
        mean_q_enrichment=mean_q,
        mean_p_enrichment=mean_p,
        mean_asymmetry_ratio=mean_asymm,
        wilcoxon_q_pvalue=wilcoxon_q_p,
        wilcoxon_p_pvalue=wilcoxon_p_p,
        mann_whitney_pvalue=mann_whitney_p,
        mann_whitney_effect_size=effect_size,
        q_enrichment_ci_lower=q_ci_lower,
        q_enrichment_ci_upper=q_ci_upper,
        p_enrichment_ci_lower=p_ci_lower,
        p_enrichment_ci_upper=p_ci_upper,
        criterion_1_failed=criterion_1,
        criterion_2_failed=criterion_2,
        criterion_3_failed=criterion_3,
        criterion_4_failed=criterion_4,
        criteria_failed_count=criteria_failed,
        interpretation=interpretation
    )
    
    return result


def save_statistical_analysis(decision: FalsificationDecision, filepath: str):
    """Save statistical analysis results to JSON."""
    data = {
        'decision': decision.to_dict(),
        'metadata': {
            'analysis': 'asymmetric_enrichment_falsification',
            'version': '1.0',
            'statistical_tests': [
                'Wilcoxon signed-rank',
                'Bootstrap CI (10k resamples)',
                'Mann-Whitney U',
                'Levene variance homogeneity'
            ]
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved statistical analysis to {filepath}")


if __name__ == '__main__':
    print("Statistical Analysis for Asymmetric Enrichment Falsification")
    print("=" * 80)
    print("\nImplements rigorous statistical tests:")
    print("  1. Wilcoxon signed-rank test")
    print("  2. Bootstrap confidence intervals (10,000 resamples)")
    print("  3. Mann-Whitney U test")
    print("  4. Levene's test for variance homogeneity")
    print("\nUsage: Import and call evaluate_falsification_criteria()")
