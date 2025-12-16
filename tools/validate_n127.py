#!/usr/bin/env python3
"""
Validate Z5D resonance scoring hypothesis using N127 (Scale 127) data.
Analyzes correlation between Geofac Amplitude and Z5D Score.
"""

import sys
import json
import numpy as np
import scipy.stats
from pathlib import Path
import matplotlib.pyplot as plt

# Add directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import run_geofac_peaks_mod
import z5d_adapter

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_FILE = DATA_DIR / "validation_n127_correlation.jsonl"
PLOT_FILE = DATA_DIR / "amplitude_vs_score_n127.png"

def analyze_correlation(num_samples=1000):
    print(f"Generating {num_samples} Geofac candidates at scale 127...")
    
    # Generate samples
    qmc_samples = np.random.rand(num_samples, 4)
    row_ids = list(range(num_samples))
    
    # Run Geofac
    results = run_geofac_peaks_mod.extract_geofac_peaks(
        row_ids, qmc_samples, scale_min=127, scale_max=127, approx=True
    )
    
    print(f"Scoring {len(results)} candidates...")
    
    data_points = []
    
    for i, r in enumerate(results):
        if "p" not in r:
            continue
            
        p_str = r["p"]
        amplitude = r["amplitude"]
        
        try:
            n_est = z5d_adapter.z5d_n_est(p_str)
            score = z5d_adapter.compute_z5d_score(p_str, n_est)
            
            data_points.append({
                "amplitude": amplitude,
                "score": score,
                "p": p_str
            })
        except Exception:
            continue
            
        if i % 100 == 0:
            print(f"Scored {i}/{len(results)}...", end="\r")
            
    print(f"Scored {len(data_points)} candidates.")
    
    # Analyze
    amplitudes = [d["amplitude"] for d in data_points]
    scores = [d["score"] for d in data_points]
    
    # Correlation
    pearson_r, p_val = scipy.stats.pearsonr(amplitudes, scores)
    
    print("\n=== N127 Correlation Analysis ===")
    print(f"Samples: {len(data_points)}")
    print(f"Pearson Correlation (Amplitude vs Score): {pearson_r:.4f}")
    print(f"P-value: {p_val:.4e}")
    
    if pearson_r < -0.1:
        print("RESULT: Negative correlation detected (Higher amplitude -> Lower/Better score).")
        if pearson_r < -0.3:
            print("STRENGTH: Moderate to Strong correlation.")
    else:
        print("RESULT: No significant negative correlation.")
        
    # Save results
    with open(RESULTS_FILE, "w") as f:
        for d in data_points:
            f.write(json.dumps(d) + "\n")
            
    print(f"Results saved to {RESULTS_FILE}")
    
    # Simple ASCII Plot
    # plt.scatter(amplitudes, scores) ... we can't show it.
    # We'll save a plot if possible, or just skip.
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(amplitudes, scores, alpha=0.5)
        plt.xlabel("Geofac Amplitude")
        plt.ylabel("Z5D Score")
        plt.title(f"Amplitude vs Score (Scale 127, N={len(data_points)})")
        plt.grid(True)
        plt.savefig(PLOT_FILE)
        print(f"Plot saved to {PLOT_FILE}")
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    analyze_correlation(num_samples=2000)
