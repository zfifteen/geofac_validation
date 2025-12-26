# p-adic Ultrametric vs Riemannian Metric GVA Experiment

## Purpose

This experiment compares two different geometric metrics for factor-finding in semiprime factorization:

1. **Baseline (Riemannian/Euclidean)**: The existing Z5D geometric resonance scoring based on Prime Number Theorem predictions. This creates a Riemannian-like manifold where candidates are scored by their deviation from expected prime positions.

2. **p-adic Ultrametric**: An alternative metric based on p-adic valuation and ultrametric distance. The p-adic metric provides a fundamentally different geometric structure that emphasizes divisibility properties and arithmetic structure.

The **goal** is to explore whether the p-adic ultrametric can provide better factor identification than traditional Euclidean/Riemannian metrics in a toy GVA (Geometric Variational Analysis) setting.

### Research Questions

- Does the p-adic ultrametric concentrate scoring around true factors better than the baseline?
- Are there semiprimes where one metric significantly outperforms the other?
- What structural properties of semiprimes make each metric more or less effective?

## Experiment Design

For each test semiprime N = p × q:

1. **Generate candidates**: Sample uniformly in a ±15% window around √N
2. **Score with baseline metric**: Use Z5D geometric resonance (PNT-based)
3. **Score with p-adic metric**: Use multi-prime p-adic ultrametric
4. **Sort and test**: Order candidates by score, test with GCD
5. **Record results**:
   - Did we find a nontrivial factor?
   - How many iterations until first factor?
   - Runtime and score statistics

### Test Cases

The experiment runs on a range of semiprimes:

- **Toy cases**: Small semiprimes (N < 10^4) for quick validation
- **Medium cases**: ~22-bit factors for realistic testing
- **RSA challenges**: Actual RSA-100 and similar challenges

## Directory Structure

```
experiments/p_adic_gva_test/
├── src/
│   ├── metric_baseline.py      # Baseline Riemannian/Z5D metric
│   ├── metric_padic.py         # p-adic ultrametric implementation
│   └── experiment_runner.py    # Main experiment orchestrator
├── results/
│   └── padic_gva_results_*.csv # Output CSV files (timestamped)
├── notebooks/                   # (Optional Jupyter notebooks for analysis)
└── README.md                    # This file
```

## How to Run

### Prerequisites

This experiment requires only Python 3 standard library. No external dependencies needed for basic functionality.

### Running the Experiment

From the repository root:

```bash
# Run using Python module syntax
python3 -m experiments.p_adic_gva_test.src.experiment_runner

# Or run directly
cd experiments/p_adic_gva_test/src
python3 experiment_runner.py
```

### Testing Individual Metrics

You can test each metric independently:

```bash
# Test baseline metric
python3 experiments/p_adic_gva_test/src/metric_baseline.py

# Test p-adic metric
python3 experiments/p_adic_gva_test/src/metric_padic.py
```

Each metric file includes a `__main__` section with simple demonstrations.

## Output Format

### CSV Results

Results are saved to `results/padic_gva_results_<timestamp>.csv` with the following fields:

| Field | Description |
|-------|-------------|
| `semiprime_name` | Name identifier (e.g., "Toy-1", "RSA-100-mini") |
| `N` | The semiprime value |
| `true_p` | Known smaller factor |
| `true_q` | Known larger factor |
| `description` | Human-readable description |
| `metric` | Which metric was used ("baseline" or "padic") |
| `num_candidates` | Number of candidates generated |
| `window_pct` | Search window size (% of √N) |
| `factor_found` | Boolean: was a nontrivial factor found? |
| `factor_value` | The factor found (if any) |
| `iterations_to_factor` | Number of candidates tested until factor found |
| `gcd_checks` | Total GCD checks performed |
| `runtime_seconds` | Wall-clock time for this search |
| `best_score` | Score of top-ranked candidate |
| `worst_score` | Score of lowest-ranked candidate |
| `total_scored` | Number of candidates successfully scored |

### Console Output

The experiment prints detailed progress:

```
================================================================================
Testing: Toy-1
N = 143
p = 11, q = 13
sqrt(N) ≈ 11
================================================================================

[1/2] Running with BASELINE (Riemannian/Z5D) metric...
  Factor found: True
  Factor: 11
  Iterations: 23
  Runtime: 0.0042s

[2/2] Running with P-ADIC ultrametric...
  Factor found: True
  Factor: 13
  Iterations: 15
  Runtime: 0.0038s
```

Plus a final summary table comparing performance.

## Implementation Notes

### Dataset Validation

All semiprimes are validated on startup with the following checks:
- **Multiplication check**: `assert N == p × q`
- **Coprimality**: `assert gcd(p, q) == 1`
- **Primality**: For small factors (< 10,000), verified using trial division

This prevents transcription errors and ensures the experiment operates on valid semiprimes. The validation runs automatically when importing `experiment_runner.py`.

### Metric Integrity (No Leakage)

**Critical design principle**: Neither metric computes `gcd(candidate, N)` or tests divisibility during scoring. This ensures:
- Scores reflect geometric/arithmetic structure, not direct factor testing
- Fair comparison between metrics
- Meaningful results (not just "which metric cheats better")

The GCD test is ONLY performed after scoring to check if top-ranked candidates are actual factors.

### Baseline Metric (metric_baseline.py)

The baseline metric is a **self-contained copy** of the existing Z5D scoring logic:

- **z5d_n_est(p)**: Estimates prime index using PNT approximation
- **z5d_predict_nth_prime(n)**: Predicts nth prime from index
- **compute_z5d_score(p)**: Log-relative deviation score
- **riemannian_distance(a, b)**: Geometric manifold distance
- **compute_gva_score(candidate, reference)**: Combined GVA scoring

Lower scores indicate better alignment with expected prime structure.

### p-adic Metric (metric_padic.py)

The p-adic metric implements arithmetic ultrametric distance:

- **padic_valuation(n, p)**: Highest power of prime p dividing n
- **padic_norm(n, p)**: p-adic norm ||n||_p = p^(-v_p(n))
- **padic_distance(a, b, p)**: Ultrametric distance d_p(a,b) = ||a-b||_p
- **multi_padic_distance()**: Weighted combination over multiple primes
- **padic_ultrametric_gva_score()**: Main scoring function

Key property: **Ultrametric inequality** d(a,c) ≤ max(d(a,b), d(b,c))

This creates a very different geometric structure than Euclidean/Riemannian metrics.

### Experiment Runner (experiment_runner.py)

Orchestrates the full experiment:

1. Defines test semiprimes
2. For each semiprime:
   - Generates uniform random candidates in search window
   - Scores with both metrics
   - Attempts factorization via GCD on top-scored candidates
   - Records all results
3. Saves results to timestamped CSV
4. Prints comparison summary

## Interpretation Guide

### Success Metrics

- **Factor found**: Did the metric guide us to a nontrivial factor?
- **Iterations to factor**: How many top-scored candidates did we need to test?
- **Runtime**: Computational cost

### What to Look For

1. **Overall success rate**: Which metric finds more factors?
2. **Efficiency**: When both succeed, which needs fewer iterations?
3. **Failure patterns**: Are there semiprimes where one metric consistently fails?
4. **Score distributions**: How well do scores correlate with factor proximity?

### Limitations

- **Small sample**: This is a toy experiment with limited test cases
- **One-factor goal**: We stop at the first factor found
- **Simple search**: No adaptive windowing or sophisticated sampling
- **Uniform random sampling limitation**: For large semiprimes like RSA-100, the probability of randomly sampling the exact factors is astronomically small (≈10^-47). The failure on RSA-100 is expected and does not indicate metric quality - it reflects the sampling strategy limitation.
- **Window coverage vs sampling density**: For RSA-100, the ±15% window around √N contains both factors, but 500 random samples cannot densely cover this space. Success on small semiprimes (N < 10^13) is due to dense sampling relative to window size.

This is an **exploratory experiment** to motivate deeper investigation, not a rigorous benchmark.

## Expected Outcomes

### Hypothesis

The baseline (Z5D/Riemannian) metric should perform well because it's been validated on geometric resonance patterns in prior work. The p-adic metric may struggle because:

- It emphasizes divisibility structure, not geometric position
- The ultrametric property doesn't naturally align with factorization distance
- Small primes may not capture enough structure for large semiprimes

However, p-adic metrics might show advantages for:

- Semiprimes with special arithmetic structure
- Cases where factors share small prime divisors
- Situations where Euclidean distance is misleading

### Null Hypothesis

If p-adic and baseline metrics perform similarly, it suggests:

- The geometric structure of factorization is robust to metric choice
- OR: Our search window contains factors for almost all test cases (metric doesn't matter)
- OR: Both metrics are essentially scoring candidates randomly

## Future Directions

If this experiment shows promise:

1. **Hybrid metrics**: Combine p-adic and Riemannian in a composite score
2. **Adaptive p-adic**: Choose primes dynamically based on N's structure
3. **Larger test suite**: Scale to RSA-200, RSA-300+ challenges
4. **Theoretical analysis**: Why does one metric outperform the other?
5. **Machine learning**: Use both metrics as features in learned models

## Contact & References

This is a self-contained experiment within the `geofac_validation` repository.

For more on the baseline Z5D metric, see:
- `z5d_adapter.py` in the repository root
- `experiments/z5d_validation_n127.py` for validation studies

For p-adic number theory:
- Gouvêa, "p-adic Numbers: An Introduction"
- Koblitz, "p-adic Numbers, p-adic Analysis, and Zeta-Functions"
