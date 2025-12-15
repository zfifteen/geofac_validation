# Factor Attempt for N127

This directory contains the scripts and configuration for attempting to factor the 127-bit semiprime `N_127`.

## Search Window Coverage

The search strategy employs a symmetric search window around $\sqrt{N_{127}}$.

- **Center**: $\sqrt{N_{127}} \approx 1.17 \times 10^{19}$
- **Radius**: $2^{42} \approx 4.4 \times 10^{12}$
- **Range**: [$\sqrt{N} - 2^{42}$, $\sqrt{N} + 2^{42}$]

This window size is chosen based on the heuristic that for balanced semiprimes (where factors are close to $\sqrt{N}$), the factors are likely to fall within this range. The specific radius $2^{42}$ corresponds to approximately $1/3$ of the bit length of $N$, providing a wide enough net while keeping the search space manageable for the given time budget.

## Configuration

The configuration parameters are defined in `config_127.py`:

- `N_127`: The target semiprime.
- `BITS`: Bit length of `N_127`.
- `SQRT_N`: Integer square root of `N_127`.
- `RADIUS`: Search radius.
- `MAX_WALLCLOCK_SECONDS`: Maximum runtime (30 minutes).
- `TOTAL_CANDIDATES`: Total number of candidates to evaluate (5,000,000).
- `NUM_BATCHES`: Number of batches to process.
- `TOP_K_PER_BATCH`: Number of top-scoring candidates to check per batch.
