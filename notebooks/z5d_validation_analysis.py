#!/usr/bin/env python3
"""
Phase 2: Spatial Distribution Analysis of Z5D Scores (N127).
Analyzes enrichment of candidates near factors in top-scored cohorts.
"""

import sys
import csv
import json
import math
import numpy as np
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_FILE = Path(__file__).parent.parent / "experiments" / "z5d_validation_n127_results.csv"
GROUND_TRUTH_FILE = DATA_DIR / "n127_ground_truth.json"

def load_ground_truth():
    with open(GROUND_TRUTH_FILE, "r") as f:
        return json.load(f)

def load_results():
    data = []
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def analyze():
    print("Loading data...")
    gt = load_ground_truth()
    results = load_results()
    
    if not results:
        print("No results found.")
        return

    # Convert to numeric
    scores = []
    dists_p = []
    dists_q = []
    
    # We need p and q values to define zones relative to magnitude?
    # Or just use the distances provided.
    
    # N is approx 10^1233. sqrt(N) approx 10^616.
    # 1% zone is 10^614.
    # Let's check the distances.
    
    # Pre-process
    processed = []
    for r in results:
        score = float(r["z5d_score"])
        dist_p = float(r["dist_p"]) # float might overflow if dist is huge?
        # dist_p is difference between 1233-digit numbers? No.
        # Candidates are around sqrt(N) (616 digits).
        # p and q are 616 digits.
        # dist_p is difference. Max difference is 13% of 10^616.
        # 10^616 is too big for float.
        # We need to handle distances as large numbers or normalized.
        
        # But wait, r["dist_p"] is a string of digits.
        # We can just check if it's "small".
        # 1% of 10^616 is 10^614.
        # If len(dist_p) < 614, it's definitely in the zone.
        # If len(dist_p) == 614, we check value.
        
        # Let's use log10(distance) for plotting/analysis if possible.
        # Or just enrichment.
        
        # Enrichment definition:
        # Zone: Within 1% of p or q.
        # 1% of sqrt(N).
        # We need sqrt(N).
        
        p_str = r["dist_p"]
        q_str = r["dist_q"]
        
        # Approximate check
        # sqrt(N) has ~616 digits.
        # 1% threshold has ~614 digits.
        
        in_p_zone = (len(p_str) <= 614)
        in_q_zone = (len(q_str) <= 614)
        
        processed.append({
            "score": score,
            "in_p_zone": in_p_zone,
            "in_q_zone": in_q_zone,
            "in_any_zone": in_p_zone or in_q_zone
        })
        
        scores.append(score)

    total_samples = len(processed)
    total_p_zone = sum(1 for x in processed if x["in_p_zone"])
    total_q_zone = sum(1 for x in processed if x["in_q_zone"])
    total_any_zone = sum(1 for x in processed if x["in_any_zone"])
    
    print(f"Total Samples: {total_samples}")
    print(f"Base Rate (P-Zone): {total_p_zone/total_samples:.4%}")
    print(f"Base Rate (Q-Zone): {total_q_zone/total_samples:.4%}")
    
    # Sort by Score (Ascending - lower is better)
    processed.sort(key=lambda x: x["score"])
    
    cohorts = [100, 500, 1000, 2000]
    
    print("\n=== Enrichment Analysis ===")
    print(f"{ 'Top-K':<10} { 'Score Threshold':<20} { 'P-Zone Enrich':<15} { 'Q-Zone Enrich':<15}")
    
    for k in cohorts:
        if k > total_samples: continue
        
        cohort = processed[:k]
        threshold = cohort[-1]["score"]
        
        k_p = sum(1 for x in cohort if x["in_p_zone"])
        k_q = sum(1 for x in cohort if x["in_q_zone"])
        
        # Enrichment = (k_zone / k) / (total_zone / total)
        # Avoid div by zero
        p_enrich = (k_p / k) / (total_p_zone / total_samples) if total_p_zone else 0
        q_enrich = (k_q / k) / (total_q_zone / total_samples) if total_q_zone else 0
        
        print(f"{k:<10} {threshold:<20.6f} {p_enrich:<15.2f} {q_enrich:<15.2f}")
        
    # Statistical Test (KS)
    # Compare scores of "In Zone" vs "Out of Zone"
    in_zone_scores = [x["score"] for x in processed if x["in_any_zone"]]
    out_zone_scores = [x["score"] for x in processed if not x["in_any_zone"]]
    
    if in_zone_scores and out_zone_scores:
        mean_in = np.mean(in_zone_scores)
        mean_out = np.mean(out_zone_scores)
        print(f"Mean Score (In-Zone): {mean_in:.6f}")
        print(f"Mean Score (Out-Zone): {mean_out:.6f}")
        
        ks_stat, p_val = stats.ks_2samp(in_zone_scores, out_zone_scores)
        print("\n=== Statistical Significance (KS Test) ===")
        print("Hypothesis: In-Zone scores distribution != Out-Zone scores distribution")
        print(f"KS Statistic: {ks_stat:.4f}")
        print(f"P-Value: {p_val:.4e}")
        
        if p_val < 0.05:
            print("RESULT: Significant difference detected.")
        else:
            print("RESULT: No significant difference.")
            
    else:
        print("\nCannot run KS test: Empty zones.")

if __name__ == "__main__":
    analyze()
