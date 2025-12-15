import math

N_127 = 137524771864208156028430259349934309717
BITS = N_127.bit_length()
SQRT_N = math.isqrt(N_127)

# Generic symmetric search window around sqrt(N), no factor knowledge
# RADIUS = 2^(BITS//3) ≈ 2^42 for 127-bit N, covering ~4.4e12 range
# For balanced 127-bit factors, expect ~2^63.5 scale, so coverage fraction ~3e-6
RADIUS = 1 << (BITS // 3)
SEARCH_MIN = max(2, SQRT_N - RADIUS)
SEARCH_MAX = SQRT_N + RADIUS

MAX_WALLCLOCK_SECONDS = 30 * 60  # 30 minutes

TOTAL_CANDIDATES = 5_000_000  # tune if too slow/fast
NUM_BATCHES = 50
TOP_K_PER_BATCH = 10_000
