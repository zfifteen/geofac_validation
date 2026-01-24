"""
Main experiment runner with all three generator strategies.
"""
import json
import time
import numpy as np
import pandas as pd
from math import isqrt
from dataclasses import asdict

from generate_test_corpus import SemiprimeCase
from qmc_candidate_generator import QMCCandidateGenerator, RandomCandidateGenerator
from z5d_score_emulator import z5d_score
from enrichment_analyzer import compute_enrichment, EnrichmentResult

def run_trial(case: SemiprimeCase, generator, scorer) -> dict:
    """Execute single trial and collect metrics."""
    start = time.time()
    sqrt_N = isqrt(case.N)  # CORRECTED: integer sqrt for large N
    candidates = generator.generate_candidates(sqrt_N)
    
    scores = [scorer(c, case.N, sqrt_N) for c in candidates]
    
    # CORRECTED: Sort ascending for more negative = better
    sorted_pairs = sorted(zip(scores, candidates))  # ascending: smallest (most negative) first
    checks = len(candidates)
    min_dist = float('inf')
    for i, (_, c) in enumerate(sorted_pairs, 1):
        dist_p = abs(c - case.p)
        dist_q = abs(c - case.q)
        current_min = min(dist_p, dist_q)
        if current_min < min_dist:
            min_dist = current_min
            checks = i  # Proxy: rank where closest-to-factor candidate appears
    enrichment = compute_enrichment(candidates, case.p, case.q, sqrt_N)
    wall_time = (time.time() - start) * 1000
    
    return {
        'N': str(case.N),
        'magnitude': case.magnitude,
        'imbalance_ratio': case.imbalance_ratio,
        'checks_to_find_factor': checks,
        'total_candidates': len(candidates),
        'enrichment_ratio': enrichment.enrichment_ratio,
        'near_p_count': enrichment.near_p_count,
        'near_q_count': enrichment.near_q_count,
        'ks_pvalue': enrichment.ks_pvalue,
        'mw_pvalue': enrichment.mw_pvalue,
        'score_variance': float(np.var(scores)),
        'wall_time_ms': wall_time
    }

def main(corpus_path: str, output_path: str):
    with open(corpus_path) as f:
        raw = json.load(f)
    corpus = [SemiprimeCase(**c) for c in raw]
    
    generators = {
        'symmetric_random': RandomCandidateGenerator(seed=42, asymmetric=False),
        'symmetric_qmc': QMCCandidateGenerator(seed=42, asymmetric=False),
        'asymmetric_qmc': QMCCandidateGenerator(seed=42, asymmetric=True)
    }
    
    results = []
    for case in corpus:
        for name, gen in generators.items():
            trial = run_trial(case, gen, z5d_score)
            trial['generator'] = name
            results.append(trial)
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results: {output_path} ({len(results)} trials)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus.json")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()
    main(args.corpus, args.output)
