#!/usr/bin/env python3
"""
Generate 20 synthetic semiprimes for Z5D adversarial testing.
"""

import random
from sympy import nextprime


def gen_semiprime(bits):
    """Generate semiprime with factors ~bits/2"""
    p = nextprime(random.getrandbits(bits // 2))
    q = nextprime(random.getrandbits(bits // 2))
    N = p * q
    return str(N), str(p), str(q)


# Generate 20 semiprimes: mix of 128, 192, 256 bits
synthetics = []
for i in range(20):
    bits = random.choice([128, 192, 256])
    N, p, q = gen_semiprime(bits)
    synthetics.append(
        {"name": f"Synthetic-{i + 1} ({bits}bit)", "N": N, "p": p, "q": q}
    )

# Print for use
import json

print(json.dumps(synthetics, indent=2))
