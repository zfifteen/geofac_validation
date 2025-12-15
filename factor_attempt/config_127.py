import math

N_127 = 137524771864208156028430259349934309717
BITS = N_127.bit_length()
SQRT_N = math.isqrt(N_127)

# Generic symmetric search window around sqrt(N), no factor knowledge
RADIUS = 1 << (BITS // 3)
SEARCH_MIN = max(2, SQRT_N - RADIUS)
SEARCH_MAX = SQRT_N + RADIUS

MAX_WALLCLOCK_SECONDS = 30 * 60  # 30 minutes

TOTAL_CANDIDATES = 5_242_880      # power of 2 for QMC (5 * 2^20)
NUM_BATCHES = 80
TOP_K_PER_BATCH = 10_000
