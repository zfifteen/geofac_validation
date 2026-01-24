"""
Generates semiprimes N = p × q with controlled imbalance ratios.
Magnitudes: 10^20, 10^30, 10^40 (feasible for sympy in <1min/trial)
Imbalance: ratio = log(q)/log(p) ∈ {1.0, 1.5, 2.0}
"""
import random
import math
import json
import time
from sympy import nextprime, isprime
from dataclasses import dataclass, asdict

@dataclass
class SemiprimeCase:
    N: int
    p: int  # smaller factor
    q: int  # larger factor
    magnitude: int
    imbalance_ratio: float

def generate_corpus(magnitudes=[20, 30, 40], 
                    ratios=[1.0, 1.5, 2.0],
                    samples_per_cell=10,
                    seed=20260123,
                    timeout_per_sample=60) -> list[SemiprimeCase]:
    """Deterministic corpus generation with pinned seed and timeout."""
    random.seed(seed)
    corpus = []
    for mag in magnitudes:
        for ratio in ratios:
            for _ in range(samples_per_cell):
                start_time = time.time()
                p_digits = int(mag / (1 + ratio))
                q_digits = mag - p_digits
                
                p_start = random.randint(10**(p_digits-1), 10**p_digits - 1)
                p = nextprime(p_start)
                
                q_start = random.randint(10**(q_digits-1), 10**q_digits - 1)
                q = nextprime(q_start)
                
                if time.time() - start_time > timeout_per_sample:
                    continue
                if p < q and isprime(p) and isprime(q):
                    N = p * q
                    actual_ratio = math.log10(q) / math.log10(p) if p > 1 else 1.0
                    corpus.append(SemiprimeCase(N, p, q, mag, round(actual_ratio, 2)))
    return corpus

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260123)
    parser.add_argument("--output", type=str, default="corpus.json")
    args = parser.parse_args()
    
    corpus = generate_corpus(seed=args.seed)
    with open(args.output, "w") as f:
        json.dump([asdict(c) for c in corpus], f, indent=2)
    print(f"Generated {len(corpus)} semiprimes → {args.output}")
