# Validation of Z5D Resonance Scoring using N127 Ground Truth

## Objective
Validate the hypothesis that Z5D resonance scoring can concentrate candidate factors near the true factors of N127 more effectively than random sampling. Specifically, we investigate whether candidates with high Geofac resonance amplitudes exhibit better (lower) Z5D scores.

## Methodology

1.  **Candidate Generation**:
    *   Generated 1,000 candidates at Scale 127 ($N 
approx 10^{127}$) using Quasi-Monte Carlo (Sobol) seeds.
    *   Used `tools/run_geofac_primes.py` to ensure candidates ($p$) are true primes (using `gmpy2.next_prime`).
    *   Computed Geofac Resonance Amplitude for each candidate using the simplified Dirichlet-style resonance scan.

2.  **Scoring**:
    *   Computed Z5D scores for all candidates using `z5d_adapter.py`.
    *   Score formula: $\text{score} = \log_{10}(|p - \text{predicted\_p}| / p)$.

3.  **Analysis**:
    *   Calculated Pearson correlation between Geofac Amplitude and Z5D Score.
    *   Compared mean scores of top-10% amplitude candidates vs bottom-10%.

## Results

*   **Sample Size**: 1,000 candidates (Scale 127)
*   **Pearson Correlation**: 0.0093 (P-value: 0.77)
*   **Mean Score (Top 100 Amplitude)**: -5.9875
*   **Mean Score (Bottom 100 Amplitude)**: -5.9875

## Conclusion

The analysis **did not find a significant correlation** between Geofac resonance amplitude and Z5D scores at Scale 127.
The correlation coefficient is near zero, and there is no difference in scores between high-amplitude and low-amplitude candidates.

This suggests that at Scale 127, the current Geofac resonance metric does not effectively filter for candidates that align better with the Z5D/PNT prediction model compared to random sampling within the same magnitude range.

## Next Steps
*   Investigate if the resonance metric needs calibration for Scale 127.
*   Increase sample size to 1,000,000 to rule out statistical noise (though current p-value is very high).
*   Verify if the "Approximation Mode" in Geofac (using `p_leading * 10^k`) introduces artifacts that mask the resonance signal.
