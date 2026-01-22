#!/usr/bin/env python3
"""
Generate Prospective Test Semiprimes for Gradient Zoom Validation

This script generates 20 fresh semiprimes with pre-specified characteristics
for prospective validation of the gradient descent algorithm. The semiprimes
are pre-registered by committing only N values (factors stored separately).

USAGE
=====
python3 generate_prospective_semiprimes.py --output prospective_semiprimes.json

OUTPUT
======
Generates two files:
1. prospective_semiprimes.json - Public N values (for testing)
2. prospective_semiprimes_factors.json - Private factors (for verification)

DATASET SPECIFICATION (Per Issue #43)
======================================
20 fresh semiprimes:

10 in 80-100 bit range:
  - 3 balanced: |p - q| < 0.05√N
  - 4 moderate: 0.10√N < |p - q| < 0.30√N
  - 3 extreme: 0.50√N < |p - q| < 1.00√N

10 in 120-140 bit range:
  - 2 balanced: |p - q| < 0.05√N
  - 3 moderate: 0.10√N < |p - q| < 0.30√N
  - 5 extreme: 0.50√N < |p - q| < 1.00√N
"""

import gmpy2
import json
import argparse
import random
from pathlib import Path
from typing import Tuple, Dict, Any


def generate_prime_near(target: int, bits: int) -> gmpy2.mpz:
    """
    Generate a prime near the target value within the specified bit range.
    
    Args:
        target: Target value
        bits: Desired bit length
    
    Returns:
        gmpy2.mpz: A prime number
    """
    # Start near target and search for prime
    candidate = gmpy2.mpz(target)
    
    # Ensure odd
    if candidate % 2 == 0:
        candidate += 1
    
    # Search for prime (limiting search to avoid infinite loop)
    max_attempts = 10000
    for _ in range(max_attempts):
        if gmpy2.is_prime(candidate):
            # Check bit length
            if candidate.bit_length() == bits:
                return candidate
        candidate += 2  # Next odd number
    
    # Fallback: use gmpy2's random prime
    return gmpy2.next_prime(gmpy2.mpz(2) ** (bits - 1))


def generate_semiprime_with_offset(target_bits: int, offset_type: str, 
                                   seed: int = None) -> Dict[str, Any]:
    """
    Generate a semiprime with specified factor offset characteristics.
    
    Args:
        target_bits: Target bit length for the semiprime
        offset_type: "balanced", "moderate", or "extreme"
        seed: Random seed for reproducibility
    
    Returns:
        dict: Semiprime metadata including N, p, q, and characteristics
    """
    if seed is not None:
        random.seed(seed)
        gmpy2.random_state(seed)
    
    # Calculate target factor bit lengths
    # For N ≈ 2^target_bits, we want p, q ≈ 2^(target_bits/2)
    factor_bits = target_bits // 2
    
    # Generate base factors using gmpy2 for arbitrary precision
    # Start with balanced factors (p ≈ q ≈ √N)
    p_base_range = gmpy2.mpz(2) ** (factor_bits - 2)
    # Use gmpy2.mpz_random for arbitrary precision random generation
    random_state = gmpy2.random_state(seed) if seed is not None else gmpy2.random_state()
    p_offset_random = gmpy2.mpz_urandomb(random_state, factor_bits - 2)
    p_base = gmpy2.mpz(2) ** (factor_bits - 1) + p_offset_random
    p = gmpy2.next_prime(p_base)
    
    # Calculate target sqrt(N) for offset calculations
    # Approximate sqrt(N) ≈ 2^(target_bits/2)
    sqrt_N_approx = gmpy2.mpz(2) ** (factor_bits)
    
    # Generate q based on offset type
    # Important: for extreme offsets, ensure q stays within reasonable bit range
    # to avoid producing semiprimes much smaller than target
    min_q_bits = max(2, factor_bits - 5)  # Allow up to 5 bit reduction
    
    if offset_type == "balanced":
        # |p - q| < 0.05√N
        max_offset = int(sqrt_N_approx * 0.05)
        q_target = p + random.randint(-max_offset, max_offset)
    elif offset_type == "moderate":
        # 0.10√N < |p - q| < 0.30√N
        min_offset = int(sqrt_N_approx * 0.10)
        max_offset = int(sqrt_N_approx * 0.30)
        offset = random.randint(min_offset, max_offset)
        q_target = p + offset if random.random() > 0.5 else p - offset
    elif offset_type == "extreme":
        # 0.50√N < |p - q| < 1.00√N
        # For extreme, prefer adding to p rather than subtracting to avoid tiny q
        min_offset = int(sqrt_N_approx * 0.50)
        max_offset = int(sqrt_N_approx * 1.00)
        offset = random.randint(min_offset, max_offset)
        # Bias toward larger factor (add offset 75% of the time)
        q_target = p + offset if random.random() > 0.25 else max(p - offset, sqrt_N_approx // 4)
    else:
        raise ValueError(f"Unknown offset_type: {offset_type}")
    
    # Ensure q_target is positive and of reasonable size
    q_target = max(gmpy2.mpz(2) ** min_q_bits, q_target)
    q = gmpy2.next_prime(q_target)
    
    # Compute semiprime
    N = p * q
    
    # Verify and compute actual characteristics
    actual_bits = N.bit_length()
    sqrt_N = gmpy2.isqrt(N)
    p_offset = abs(int(p) - int(sqrt_N))
    q_offset = abs(int(q) - int(sqrt_N))
    p_offset_pct = (p_offset / float(sqrt_N)) * 100
    q_offset_pct = (q_offset / float(sqrt_N)) * 100
    
    return {
        "N": str(N),
        "p": str(p),
        "q": str(q),
        "bits": actual_bits,
        "target_bits": target_bits,
        "offset_type": offset_type,
        "sqrt_N": str(sqrt_N),
        "p_offset": p_offset,
        "q_offset": q_offset,
        "p_offset_pct": p_offset_pct,
        "q_offset_pct": q_offset_pct,
        "p_bits": p.bit_length(),
        "q_bits": q.bit_length()
    }


def generate_prospective_dataset(base_seed: int = 42) -> Tuple[list, list]:
    """
    Generate the complete prospective validation dataset.
    
    Args:
        base_seed: Base random seed for reproducibility
    
    Returns:
        tuple: (public_data, private_data) lists of dicts
    """
    public_data = []
    private_data = []
    
    specifications = [
        # 80-100 bit range (10 semiprimes)
        (90, "balanced", "80-100_balanced_1"),
        (95, "balanced", "80-100_balanced_2"),
        (100, "balanced", "80-100_balanced_3"),
        (85, "moderate", "80-100_moderate_1"),
        (90, "moderate", "80-100_moderate_2"),
        (95, "moderate", "80-100_moderate_3"),
        (100, "moderate", "80-100_moderate_4"),
        (85, "extreme", "80-100_extreme_1"),
        (95, "extreme", "80-100_extreme_2"),
        (100, "extreme", "80-100_extreme_3"),
        
        # 120-140 bit range (10 semiprimes)
        (125, "balanced", "120-140_balanced_1"),
        (135, "balanced", "120-140_balanced_2"),
        (122, "moderate", "120-140_moderate_1"),
        (130, "moderate", "120-140_moderate_2"),
        (138, "moderate", "120-140_moderate_3"),
        (124, "extreme", "120-140_extreme_1"),
        (128, "extreme", "120-140_extreme_2"),
        (132, "extreme", "120-140_extreme_3"),
        (136, "extreme", "120-140_extreme_4"),
        (140, "extreme", "120-140_extreme_5"),
    ]
    
    for idx, (target_bits, offset_type, name) in enumerate(specifications):
        seed = base_seed + idx
        print(f"Generating {name}: {target_bits}-bit, {offset_type}...")
        
        data = generate_semiprime_with_offset(target_bits, offset_type, seed=seed)
        
        # Public data (N only)
        public_data.append({
            "id": name,
            "N": data["N"],
            "bits": data["bits"],
            "offset_type": data["offset_type"],
            "target_bits": data["target_bits"]
        })
        
        # Private data (full details)
        private_data.append({
            "id": name,
            **data
        })
        
        print(f"  ✓ N = {data['N'][:20]}... ({data['bits']} bits)")
        print(f"    p offset: {data['p_offset_pct']:.2f}%, q offset: {data['q_offset_pct']:.2f}%")
    
    return public_data, private_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate prospective test semiprimes for gradient zoom validation"
    )
    parser.add_argument(
        "--output",
        default="prospective_semiprimes.json",
        help="Output file for public N values"
    )
    parser.add_argument(
        "--factors-output",
        default="prospective_semiprimes_factors.json",
        help="Output file for private factors"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PROSPECTIVE SEMIPRIME GENERATION")
    print("=" * 80)
    print(f"Random seed: {args.seed}")
    print(f"Public output: {args.output}")
    print(f"Private output: {args.factors_output}")
    print("=" * 80)
    print()
    
    # Generate dataset
    public_data, private_data = generate_prospective_dataset(base_seed=args.seed)
    
    # Save public data
    with open(args.output, 'w') as f:
        json.dump(public_data, f, indent=2)
    print(f"\n✓ Saved public N values to: {args.output}")
    
    # Save private data
    with open(args.factors_output, 'w') as f:
        json.dump(private_data, f, indent=2)
    print(f"✓ Saved private factors to: {args.factors_output}")
    
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total semiprimes: {len(public_data)}")
    print(f"\nBy bit range:")
    print(f"  80-100 bits: {sum(1 for d in public_data if 80 <= d['bits'] <= 100)}")
    print(f"  120-140 bits: {sum(1 for d in public_data if 120 <= d['bits'] <= 140)}")
    print(f"\nBy offset type:")
    for offset_type in ["balanced", "moderate", "extreme"]:
        count = sum(1 for d in public_data if d['offset_type'] == offset_type)
        print(f"  {offset_type}: {count}")
    print("=" * 80)
    
    print("\n⚠️  IMPORTANT: Keep prospective_semiprimes_factors.json SECRET!")
    print("Only prospective_semiprimes.json should be committed to the repository.")
    print("\nAdd to .gitignore:")
    print("  prospective_semiprimes_factors.json")


if __name__ == "__main__":
    main()
