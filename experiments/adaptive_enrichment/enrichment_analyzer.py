"""
Measures asymmetric enrichment near p vs q.
CORRECTED: Tests distance distributions, not candidate subsets.
"""
from scipy.stats import ks_2samp, mannwhitneyu
from dataclasses import dataclass
import numpy as np

@dataclass
class EnrichmentResult:
    near_p_count: int
    near_q_count: int
    enrichment_ratio: float
    ks_statistic: float
    ks_pvalue: float
    mw_statistic: float
    mw_pvalue: float
    mean_dist_to_p: float
    mean_dist_to_q: float

def compute_enrichment(candidates: list[int],
                       p: int, q: int, sqrt_N: int,
                       proximity_threshold: float = 0.05) -> EnrichmentResult:
    """
    CORRECTED: Compare distributions of distances to p vs distances to q.
    H₁: candidates are systematically closer to q (dist_to_q < dist_to_p)
    """
    candidates = np.array(candidates, dtype=np.float64)
    
    # Compute relative distances to each factor
    dist_to_p = np.abs(candidates - p) / p
    dist_to_q = np.abs(candidates - q) / q
    
    # Count candidates within proximity threshold
    near_p_count = int(np.sum(dist_to_p < proximity_threshold))
    near_q_count = int(np.sum(dist_to_q < proximity_threshold))
    
    # Enrichment ratio (guarded against division by zero)
    enrichment_ratio = near_q_count / near_p_count if near_p_count > 0 else float('inf')
    
    # CORRECTED: KS test compares distance-to-p vs distance-to-q distributions
    if len(candidates) >= 5:
        ks_stat, ks_p = ks_2samp(dist_to_q, dist_to_p, alternative='less')
        mw_stat, mw_p = mannwhitneyu(dist_to_q, dist_to_p, alternative='less')
    else:
        ks_stat, ks_p = 0.0, 1.0
        mw_stat, mw_p = 0.0, 1.0
    
    return EnrichmentResult(
        near_p_count=near_p_count,
        near_q_count=near_q_count,
        enrichment_ratio=enrichment_ratio,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        mw_statistic=mw_stat,
        mw_pvalue=mw_p,
        mean_dist_to_p=float(np.mean(dist_to_p)),
        mean_dist_to_q=float(np.mean(dist_to_q))
    )
