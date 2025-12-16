#!/usr/bin/env python3
"""
Z5D Validation Analysis for N₁₂₇

Jupyter-style notebook for analyzing Z5D resonance scoring hypothesis.
Loads results from experiments/z5d_validation_n127_results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import gmpy2
import sys
from pathlib import Path

# Load data with error handling
csv_path = (
    Path(__file__).parent.parent / "experiments" / "z5d_validation_n127_results.csv"
)
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Keep large integers as strings for arbitrary precision
# Only convert floats to numeric
df["candidate"] = df["candidate"].astype(str)
df["z5d_score"] = pd.to_numeric(df["z5d_score"])
df["distance_to_p"] = df["distance_to_p"].astype(str)
df["distance_to_q"] = df["distance_to_q"].astype(str)
df["percent_from_sqrt"] = pd.to_numeric(df["percent_from_sqrt"])

print("Data loaded. Shape:", df.shape)
print(df.head())

# Ground truth positions
P_PERCENT = -10.3902293058
Q_PERCENT = 11.5949736567

# Phase 2: Analyze spatial distribution

# Rank by Z5D score (ascending, lower score is better)
df_sorted = df.sort_values("z5d_score").reset_index(drop=True)
df_sorted["min_dist_to_factor_mpz"] = df_sorted.apply(
    lambda row: min(gmpy2.mpz(row["distance_to_p"]), gmpy2.mpz(row["distance_to_q"])),
    axis=1,
)
df_sorted["min_dist_to_factor"] = df_sorted["min_dist_to_factor_mpz"].apply(float)


# Function to compute enrichment
def compute_enrichment(df_sorted, top_k, target_percent_range):
    """Compute enrichment factor for top-K in target zone"""

    top_k_df = df_sorted.head(top_k)

    # Random baseline: uniform in percent_from_sqrt
    percent_min = df["percent_from_sqrt"].min()
    percent_max = df["percent_from_sqrt"].max()
    target_width = target_percent_range[1] - target_percent_range[0]

    random_baseline = target_width / (percent_max - percent_min)

    # Actual in top-K within target
    in_target = top_k_df[
        (top_k_df["percent_from_sqrt"] >= target_percent_range[0])
        & (top_k_df["percent_from_sqrt"] <= target_percent_range[1])
    ]

    actual_percent = len(in_target) / top_k

    enrichment = actual_percent / random_baseline

    return enrichment, actual_percent, random_baseline


# Target zones
targets = {
    "p_zone": (-12, -9),  # around p's -10.39%
    "q_zone": (9, 14),  # around q's 11.59%
    "combined_zone": (-5, 5),  # within 5% of factors (but factors are outside)
    # Wait, better: near p or q
    "near_factors": (-12, 14),  # broad zone
}

# Compute for Top-K
top_ks = [100, 1000, 10000, 100000]

results = {}
for top_k in top_ks:
    results[top_k] = {}
    for zone_name, zone_range in targets.items():
        enrichment, actual, baseline = compute_enrichment(df_sorted, top_k, zone_range)
        results[top_k][zone_name] = {
            "enrichment": enrichment,
            "actual_percent": actual,
            "baseline_percent": baseline,
        }
        print(
            f"Top-{top_k}, {zone_name}: enrichment={enrichment:.2f}, actual={actual:.4f}, baseline={baseline:.4f}"
        )

# Plots

# 1. Histogram: Z5D score vs distance from sqrt(N)
plt.figure(figsize=(10, 6))
plt.hist2d(df["percent_from_sqrt"], df["z5d_score"], bins=50, cmap="viridis")
plt.colorbar(label="Count")
plt.xlabel("Percent from sqrt(N)")
plt.ylabel("Z5D Score")
plt.title("Z5D Score vs Distance from sqrt(N)")
plt.axvline(P_PERCENT, color="red", linestyle="--", label=f"p at {P_PERCENT:.1f}%")
plt.axvline(Q_PERCENT, color="blue", linestyle="--", label=f"q at {Q_PERCENT:.1f}%")
plt.legend()
plt.savefig(Path(__file__).parent.parent / "docs" / "z5d_score_vs_distance.png")
plt.show()

# 2. Scatter: Z5D score vs min distance to factor
# Add min distance using arbitrary precision
df["min_dist_to_factor_mpz"] = df.apply(
    lambda row: min(gmpy2.mpz(row["distance_to_p"]), gmpy2.mpz(row["distance_to_q"])),
    axis=1,
)
# For plotting/stats, convert to float (acknowledging precision loss for values > 2^53)
df["min_dist_to_factor"] = df["min_dist_to_factor_mpz"].apply(float)
plt.figure(figsize=(10, 6))
plt.scatter(df["min_dist_to_factor"], df["z5d_score"], alpha=0.1, s=1)
plt.xlabel("Min Distance to Factor")
plt.ylabel("Z5D Score")
plt.title("Z5D Score vs Min Distance to Factor")
plt.yscale("log")  # since distances are huge
plt.savefig(Path(__file__).parent.parent / "docs" / "z5d_score_vs_min_dist.png")
plt.show()

# 3. Enrichment plot
enrichments = {k: results[k]["near_factors"]["enrichment"] for k in top_ks}
plt.figure(figsize=(8, 5))
plt.plot(list(enrichments.keys()), list(enrichments.values()), marker="o")
plt.xscale("log")
plt.xlabel("Top-K")
plt.ylabel("Enrichment Factor")
plt.title("Enrichment Factor for Top-K (near factors zone)")
plt.axhline(1, color="red", linestyle="--", label="No enrichment")
plt.legend()
plt.savefig(Path(__file__).parent.parent / "docs" / "enrichment_factors.png")
plt.show()

# Phase 3: Statistical validation

# KS test: compare top-1000 spatial distribution vs random
top_1000 = df_sorted.head(1000)
ks_stat, p_value = stats.ks_2samp(
    top_1000["percent_from_sqrt"], df["percent_from_sqrt"]
)
print(f"KS test Top-1000 vs random: stat={ks_stat:.4f}, p={p_value:.2e}")

# Mann-Whitney: distances for high-Z5D vs random
# Note: min_dist_to_factor converted to float, which loses precision for values > 2^53
# This may affect exact ordering, but relative differences are preserved for statistical purposes
random_sample = df.sample(1000, random_state=42)
mann_stat, mann_p = stats.mannwhitneyu(
    top_1000["min_dist_to_factor"],
    random_sample["min_dist_to_factor"],
    alternative="less",
)
print(
    f"Mann-Whitney distances Top-1000 vs random: stat={mann_stat:.2f}, p={mann_p:.2e}"
)


# Bootstrap CI for enrichment (top-1000)
def bootstrap_enrichment(df, top_k, zone_range, n_boot=1000):
    enrichments = []
    for _ in range(n_boot):
        boot_df = df.sample(len(df), replace=True)
        boot_sorted = boot_df.sort_values("z5d_score").reset_index(drop=True)
        enrich, _, _ = compute_enrichment(boot_sorted, top_k, zone_range)
        enrichments.append(enrich)
    return np.percentile(enrichments, [2.5, 97.5])


ci = bootstrap_enrichment(df, 1000, targets["near_factors"])
print(f"Bootstrap 95% CI for enrichment (top-1000): {ci[0]:.2f} - {ci[1]:.2f}")

# Summary
print("\n=== SUMMARY ===")
print(f"Total candidates: {len(df)}")
print(f"Top-1000 enrichment: {results[1000]['near_factors']['enrichment']:.2f}")
print(f"KS p-value: {p_value:.2e}")
print(f"Mann-Whitney p-value: {mann_p:.2e}")
if results[1000]["near_factors"]["enrichment"] > 5 and p_value < 0.001:
    print("STRONG SIGNAL: Z5D provides excellent guidance!")
elif results[1000]["near_factors"]["enrichment"] > 2 and p_value < 0.05:
    print("WEAK SIGNAL: Z5D shows promise but needs refinement.")
else:
    print("NO SIGNAL: Z5D does not help for factorization.")
