#!/usr/bin/env python3
"""
Configuration constants for N_127 blind factorization attempt.

N_127 = 137524771864208156028430259349934309717 (127-bit semiprime)

All parameters tuned for a 30-minute wall-clock attempt using the existing
geofac_validation QMC and Z5D scoring pipeline.
"""

import gmpy2

# Target semiprime
N_127 = gmpy2.mpz("137524771864208156028430259349934309717")

# Mathematical constants
BIT_LENGTH = 127
SQRT_N = gmpy2.isqrt(N_127)  # Integer square root

# Search window: symmetric around sqrt(N)
# Use ~10% of sqrt(N) as search radius to cover likely factor range
SEARCH_RADIUS = SQRT_N // 10
SEARCH_MIN = SQRT_N - SEARCH_RADIUS
SEARCH_MAX = SQRT_N + SEARCH_RADIUS

# Wall-clock limit (30 minutes = 1800 seconds)
MAX_WALLCLOCK_SECONDS = 1800

# QMC candidate generation parameters
TOTAL_CANDIDATES = 5_000_000  # 5M candidates to generate
NUM_BATCHES = 50              # Process in 50 batches
BATCH_SIZE = TOTAL_CANDIDATES // NUM_BATCHES  # 100K per batch

# Scoring and filtering
TOP_K_PER_BATCH = 10_000      # Keep top 10K candidates per batch by resonance score

# QMC parameters
QMC_DIMENSIONS = 5            # Standard Z5D dimensions
QMC_SEED = 127                # Seed based on target bit length
