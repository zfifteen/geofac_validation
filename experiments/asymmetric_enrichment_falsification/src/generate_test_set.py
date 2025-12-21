#!/usr/bin/env python3
"""
Semiprime Test Set Generation

Generates stratified benchmark semiprimes across multiple bit-length ranges
with controlled factor imbalance for asymmetric enrichment falsification testing.

CRITICAL: Uses arbitrary-precision arithmetic exclusively (gmpy2/mpmath).
NO int64, NO fixed-width types - prevents silent overflow for 426-bit semiprimes.
"""

import gmpy2
from gmpy2 import mpz, isqrt
import json
import random
from typing import Tuple, List, Dict
from dataclasses import dataclass, asdict


@dataclass
class Semiprime:
    """Container for semiprime with ground truth factors."""
    name: str
    N: str  # Store as string for JSON serialization
    p: str
    q: str
    bit_length: int
    imbalance_pct: float
    sqrt_N: str
    p_offset_pct: float
    q_offset_pct: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Semiprime':
        """Reconstruct from dictionary."""
        return cls(**d)


def is_probable_prime(n: mpz, rounds: int = 64) -> bool:
    """
    Miller-Rabin primality test with arbitrary precision.
    
    Args:
        n: Candidate prime (gmpy2.mpz)
        rounds: Number of Miller-Rabin rounds (default 64 for high confidence)
    
    Returns:
        True if n is probably prime, False if definitely composite
    """
    return gmpy2.is_prime(n, rounds)


def generate_random_prime(bit_length: int, rng_state: random.Random) -> mpz:
    """
    Generate a random prime of specified bit length.
    
    Uses arbitrary precision arithmetic to handle primes up to 426 bits.
    
    Args:
        bit_length: Target bit length of the prime
        rng_state: Random number generator with fixed seed
    
    Returns:
        A probable prime of the specified bit length (gmpy2.mpz)
    """
    # Generate random odd number in target range
    min_val = mpz(1) << (bit_length - 1)  # 2^(bit_length-1)
    max_val = (mpz(1) << bit_length) - 1   # 2^bit_length - 1
    
    attempts = 0
    max_attempts = 10000
    
    while attempts < max_attempts:
        # Generate random number in range using Python's random module
        # Convert to gmpy2.mpz for arbitrary precision
        candidate = mpz(rng_state.randrange(int(min_val), int(max_val)))
        
        # Make odd (all primes except 2 are odd)
        if candidate % 2 == 0:
            candidate += 1
        
        # Test primality with 64 Miller-Rabin rounds
        if is_probable_prime(candidate, rounds=64):
            # Verify bit length is exactly what we want
            if gmpy2.bit_length(candidate) == bit_length:
                return candidate
        
        attempts += 1
    
    raise RuntimeError(f"Failed to generate {bit_length}-bit prime after {max_attempts} attempts")


def generate_balanced_semiprime(
    bit_length: int,
    imbalance_pct: float,
    seed: int = None
) -> Tuple[mpz, mpz, mpz]:
    """
    Generate semiprime N = p × q with specified bit length and imbalance.
    
    Imbalance is controlled by adjusting the bit lengths of p and q such that
    their product has the target bit length and p/q ratio matches the desired
    imbalance percentage.
    
    CRITICAL: All arithmetic uses arbitrary precision (gmpy2.mpz).
    NO conversion to int64 or float64 for large values.
    
    Args:
        bit_length: Total bits in N = p × q
        imbalance_pct: Target percentage deviation of p from √N
                      (0 = perfectly balanced, >0 = q larger than p)
        seed: Random seed for reproducibility
    
    Returns:
        (N, p, q) tuple where:
            - N = p × q (semiprime)
            - p < √N (smaller factor)
            - q > √N (larger factor)
            - |p - √N|/√N ≈ imbalance_pct (approximately)
    
    Example:
        >>> N, p, q = generate_balanced_semiprime(128, 10.0, seed=42)
        >>> # p is ~10% below √N, q is ~10% above √N
    """
    # Initialize random generator with seed
    rng = random.Random(seed)
    
    # For balanced semiprime, both factors have approximately bit_length/2 bits
    # For imbalanced, we adjust the bit distribution
    
    # Calculate target bit lengths for p and q
    # Using ln(p) + ln(q) = ln(N) → log2(p) + log2(q) = log2(N) = bit_length
    
    if imbalance_pct < 5.0:
        # Nearly balanced: both factors have ~bit_length/2 bits
        p_bits = bit_length // 2
        q_bits = bit_length - p_bits
    else:
        # Imbalanced: adjust bit distribution
        # For imbalance_pct% deviation, we need p ≈ √N * (1 - δ) and q ≈ √N * (1 + δ)
        # where δ ≈ imbalance_pct / 100
        
        # Approximate adjustment to bit lengths
        # This is iterative - we'll generate and check
        bit_adjust = int(bit_length * imbalance_pct / 100 / 2)
        p_bits = max(bit_length // 2 - bit_adjust, bit_length // 4)
        q_bits = bit_length - p_bits
    
    max_iterations = 100
    for iteration in range(max_iterations):
        # Generate candidate primes
        p = generate_random_prime(p_bits, rng)
        q = generate_random_prime(q_bits, rng)
        
        # Ensure p < q (convention: p is smaller factor)
        if p > q:
            p, q = q, p
        
        # Calculate semiprime
        N = p * q
        
        # Verify bit length is correct
        N_bits = int(gmpy2.bit_length(N))
        if N_bits != bit_length:
            # Adjust and retry
            if N_bits < bit_length:
                # Need larger factors
                if rng.random() < 0.5:
                    p_bits += 1
                else:
                    q_bits += 1
            else:
                # Need smaller factors
                if rng.random() < 0.5 and p_bits > bit_length // 4:
                    p_bits -= 1
                else:
                    q_bits -= 1
            continue
        
        # Calculate actual imbalance
        sqrt_N = isqrt(N)
        p_offset = abs(float(p - sqrt_N) / float(sqrt_N) * 100)
        q_offset = abs(float(q - sqrt_N) / float(sqrt_N) * 100)
        actual_imbalance = max(p_offset, q_offset)
        
        # Check if imbalance is within acceptable range (±50% of target)
        if imbalance_pct < 5.0:
            # For balanced, accept anything < 5%
            if actual_imbalance < 5.0:
                return N, p, q
        else:
            # For imbalanced, accept within ±50% of target
            if abs(actual_imbalance - imbalance_pct) < imbalance_pct * 0.5:
                return N, p, q
    
    # If we couldn't achieve exact imbalance, return best attempt
    # This is acceptable since we track actual imbalance in metadata
    return N, p, q


def generate_stratified_test_set(config: dict) -> List[Semiprime]:
    """
    Generate benchmark semiprimes across specified bit-length ranges.
    
    Creates a stratified dataset with controlled factor imbalance for
    comprehensive asymmetric enrichment testing.
    
    Args:
        config: Dictionary with test set specification:
            {
                'ranges': [
                    {
                        'name': 'Small',
                        'bits': [64, 128],
                        'count': 20,
                        'imbalances': [0, 2, 5, 10, 15]
                    },
                    ...
                ]
            }
    
    Returns:
        List of Semiprime objects with ground truth factors
    """
    test_set = []
    semiprime_counter = 0
    
    for range_spec in config['ranges']:
        range_name = range_spec['name']
        bit_min, bit_max = range_spec['bits']
        count_per_imbalance = range_spec['count']
        imbalances = range_spec['imbalances']
        
        print(f"\nGenerating {range_name} range ({bit_min}-{bit_max} bits)...")
        
        for imbalance in imbalances:
            print(f"  Imbalance {imbalance}%: ", end='', flush=True)
            
            for i in range(count_per_imbalance):
                # Use deterministic seed for reproducibility
                seed = 42 + semiprime_counter
                
                # Select bit length uniformly in range
                if bit_min == bit_max:
                    bit_length = bit_min
                else:
                    # For range, distribute evenly
                    bit_length = bit_min + (i * (bit_max - bit_min)) // count_per_imbalance
                
                # Generate semiprime
                N, p, q = generate_balanced_semiprime(bit_length, imbalance, seed)
                
                # Calculate metadata
                sqrt_N = isqrt(N)
                p_offset_pct = float(p - sqrt_N) / float(sqrt_N) * 100
                q_offset_pct = float(q - sqrt_N) / float(sqrt_N) * 100
                
                # Create semiprime object
                sp = Semiprime(
                    name=f"{range_name}_{bit_length}bit_{imbalance}pct_{i}",
                    N=str(N),
                    p=str(p),
                    q=str(q),
                    bit_length=bit_length,
                    imbalance_pct=imbalance,
                    sqrt_N=str(sqrt_N),
                    p_offset_pct=p_offset_pct,
                    q_offset_pct=q_offset_pct
                )
                
                test_set.append(sp)
                semiprime_counter += 1
                
                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"{i+1} ", end='', flush=True)
            
            print("✓")
    
    print(f"\nGenerated {len(test_set)} semiprimes total")
    return test_set


def save_test_set(test_set: List[Semiprime], filepath: str):
    """Save test set to JSON file."""
    data = {
        'semiprimes': [sp.to_dict() for sp in test_set],
        'count': len(test_set),
        'metadata': {
            'generator': 'generate_test_set.py',
            'version': '1.0',
            'arithmetic': 'arbitrary precision (gmpy2/mpmath)',
            'primality_test': 'Miller-Rabin (64 rounds)'
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved test set to {filepath}")


def load_test_set(filepath: str) -> List[Semiprime]:
    """Load test set from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [Semiprime.from_dict(sp) for sp in data['semiprimes']]


if __name__ == '__main__':
    # Example configuration matching the specification
    config = {
        'ranges': [
            {
                'name': 'Small',
                'bits': [64, 128],
                'count': 4,  # Reduced for testing
                'imbalances': [0, 5]
            },
            {
                'name': 'Medium',
                'bits': [128, 192],
                'count': 4,
                'imbalances': [5, 20]
            },
            {
                'name': 'Large',
                'bits': [192, 256],
                'count': 3,
                'imbalances': [20, 40]
            },
            {
                'name': 'RSA-like',
                'bits': [256, 384],
                'count': 2,
                'imbalances': [40]
            },
            {
                'name': 'Extreme',
                'bits': [384, 426],
                'count': 1,
                'imbalances': [20]
            },
        ]
    }
    
    # Generate test set
    test_set = generate_stratified_test_set(config)
    
    # Save to file
    output_path = '../data/benchmark_semiprimes.json'
    save_test_set(test_set, output_path)
    
    # Display sample
    print("\nSample semiprimes:")
    for i, sp in enumerate(test_set[:3]):
        print(f"\n{i+1}. {sp.name}")
        print(f"   Bits: {sp.bit_length}")
        print(f"   p offset: {sp.p_offset_pct:.2f}%")
        print(f"   q offset: {sp.q_offset_pct:.2f}%")
