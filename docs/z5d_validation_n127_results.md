# Z5D Validation Results (N127)

## Executive Summary
This experiment validated the Z5D resonance scoring hypothesis on a generic 1233-digit semiprime ($N_{1233}$). The hypothesis states that candidates with better (lower) Z5D scores should be spatially concentrated near the true prime factors of $N$.

**Result**: The hypothesis was **not supported** for the generic case. 
- **Enrichment**: Top-scoring candidates showed **0% enrichment** in the zones near factors (Top-2000 candidates contained 0 candidates within 1% of factors).
- **Statistical Significance**: A Kolmogorov-Smirnov test confirmed the score distributions are significantly different ($p < 10^{-200}$) between "Near-Factor" and "Far-Factor" zones, but the "Near-Factor" scores were slightly *worse* on average.

## Methodology

1.  **Ground Truth**: Generated a random 1233-digit semiprime $N = p \times q$ (Scale $10^{1233}$).
2.  **Sampling**: Generated 10,000 odd candidates uniformly distributed in a $\pm 13\%$ window around $\sqrt{N}$.
3.  **Scoring**: Computed Z5D scores for all candidates using the `z5d_adapter`.
4.  **Analysis**:
    - Defined "P-Zone" and "Q-Zone" as the range within $\pm 1\%$ of $p$ and $q$ respectively.
    - Calculated Enrichment Factor = (Fraction in Zone for Top-K) / (Base Rate).
    - Performed KS test to compare score distributions.

## detailed Results

### Enrichment Analysis
| Top-K | Score Threshold | P-Zone Enrichment | Q-Zone Enrichment |
|-------|-----------------|-------------------|-------------------|
| 100   | -8.840805       | 0.00              | 0.00              |
| 500   | -8.840799       | 0.00              | 0.00              |
| 1000  | -8.840792       | 0.00              | 0.00              |
| 2000  | -8.840777       | 0.00              | 0.00              |

*Note: Base rate for zones was ~8.1%. An enrichment of 0.00 implies strong depletion.*

### Statistical Tests
- **KS Statistic**: 0.5090
- **P-Value**: ~0 ( < 1e-200 )
- **Mean Score (In-Zone)**: -8.840733
- **Mean Score (Out-Zone)**: -8.840730

## Interpretation
The Z5D score, which measures alignment with the Prime Number Theorem prediction $p_n \approx n \ln n$, does not appear to preferentially identify the specific factors of a *randomly generated* semiprime at this scale. The significant difference in distributions suggests the metric *responds* to proximity, but not in the desired direction (candidates near factors had slightly less "perfect" PNT alignment than the best candidates elsewhere).

This contradicts findings from PR #17 (which found 5x enrichment in Q-Zone), suggesting that the specific "N127" used in that experiment may have special properties (e.g., resonance) not present in a generic semiprime, or that Z5D scoring requires calibration for generic cases.
