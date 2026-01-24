"""
PNT-based scoring without binary submodule dependency.
Implements π(x) approximation from z5d_adapter.py:45-55
"""
import math
import numpy as np

def prime_count_approx(x: int) -> float:
    """Dusart's improved PNT bound."""
    if x < 2:
        return 0.0
    ln_x = math.log(x)
    if ln_x < 1:
        return 0.0
    return (x / ln_x) * (1 + 1/ln_x + 2/(ln_x**2))

def z5d_score(candidate: int, N: int, sqrt_N: int) -> float:
    """
    Geometric resonance score proxy using PNT tightening.
    Returns log-scale score (more negative = better fit).
    """
    if candidate <= 1 or candidate >= N:
        return 0.0
    
    # Normalized distance from sqrt(N)
    dist = abs(candidate - sqrt_N) / max(sqrt_N, 1)
    
    # PNT-based density weighting
    pi_approx = prime_count_approx(candidate)
    density_factor = pi_approx / max(candidate, 1)
    
    # Combined score: subtraction creates negative scores (lower = better)
    # Near sqrt(N) with high prime density → more negative score
    score = np.log1p(dist) - np.log1p(density_factor + 1e-10)
    return score
