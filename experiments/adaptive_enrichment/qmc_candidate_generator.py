"""
Low-discrepancy Sobol sequence for resonance candidate generation.
Implements variance reduction per L'Ecuyer (2014) methodology.
"""
from scipy.stats.qmc import Sobol
import numpy as np

class QMCCandidateGenerator:
    def __init__(self, dimension: int = 1, seed: int = 42, asymmetric: bool = True):
        self.sobol = Sobol(d=dimension, scramble=True, seed=seed)
        self.asymmetric = asymmetric
        
    def generate_candidates(self, 
                            sqrt_N: int,
                            window_below: float = 0.3,
                            window_above: float = 1.0,
                            n_candidates: int = 512) -> list[int]:
        """
        Asymmetric window: [sqrt(N) - 0.3*delta, sqrt(N) + 1.0*delta]
        Symmetric window: [sqrt(N) - delta, sqrt(N) + delta]
        """
        delta = max(1, int(float(sqrt_N) ** 0.25 * 100))  # CORRECTED: float cast for large int
        if self.asymmetric:
            search_min = max(2, int(sqrt_N - delta * window_below))
            search_max = int(sqrt_N + delta * window_above)
        else:
            search_min = max(2, int(sqrt_N - delta))
            search_max = int(sqrt_N + delta)
        points = self.sobol.random(n_candidates)
        candidates = np.round(points[:, 0] * (search_max - search_min) + search_min).astype(int)
        return list(np.unique(candidates))

class RandomCandidateGenerator:
    """Baseline PRN generator for comparison."""
    def __init__(self, seed: int = 42, asymmetric: bool = False):
        self.rng = np.random.default_rng(seed)
        self.asymmetric = asymmetric
        
    def generate_candidates(self, sqrt_N: int, 
                            window_below: float = 0.3,
                            window_above: float = 1.0,
                            n_candidates: int = 512) -> list[int]:
        delta = max(1, int(float(sqrt_N) ** 0.25 * 100))
        if self.asymmetric:
            search_min = max(2, int(sqrt_N - delta * window_below))
            search_max = int(sqrt_N + delta * window_above)
        else:
            search_min = max(2, int(sqrt_N - delta))
            search_max = int(sqrt_N + delta)
        candidates = self.rng.integers(search_min, search_max + 1, size=n_candidates)
        return list(np.unique(candidates))
