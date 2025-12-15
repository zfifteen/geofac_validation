import math

N_127 = 137524771864208156028430259349934309717
BITS = N_127.bit_length()
SQRT_N = math.isqrt(N_127)

# Search window: symmetric around sqrt(N)
# Radius derived from bit length heuristic (approx 1/3 of bits)
RADIUS = 1 << (BITS // 3)
SEARCH_MIN = max(2, SQRT_N - RADIUS)
SEARCH_MAX = SQRT_N + RADIUS

# Execution constraints
MAX_WALLCLOCK_SECONDS = 30 * 60  # 30 minutes

# Search volume tuning
# 5 million candidates fits comfortably within 30 min on standard hardware
TOTAL_CANDIDATES = 5_000_000 
NUM_BATCHES = 50
TOP_K_PER_BATCH = 10_000