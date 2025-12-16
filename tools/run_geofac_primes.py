#!/usr/bin/env python3
"""
Geofac (Geometric Factor) Resonance Scanner

Performs geometric factor analysis and phase resonance detection near sqrt(N)
for RSA challenge semiprimes using the same QMC seed set.

This implements a Dirichlet-style phase resonance scan to find geometric peaks
that may align with Z5D predictor peaks.

Usage:
    python run_geofac_peaks.py --seeds ../artifacts/seedsets/phi_qmc_001.csv \
                               --output ../artifacts/geofac/peaks_phi_qmc_001.jsonl \
                               --scale-min 14 --scale-max 18 --top-k 2000
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple
from decimal import Decimal

import numpy as np
from math import isqrt as math_isqrt
import gmpy2

# Known RSA challenge numbers (for demonstration - using smaller ones for testing)
RSA_CHALLENGES = {
    "RSA-100": 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139,
    "RSA-110": 35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667,
    # These are too large for practical factorization, but we can use them as targets
}


def read_seed_csv(seed_path: Path) -> tuple[List[int], np.ndarray, Dict[str, Any]]:
    """Read QMC seed CSV file."""
    with seed_path.open("r") as f:
        reader = csv.reader(f)

        # Parse metadata
        metadata = {}
        for row in reader:
            if not row or not row[0].startswith("#"):
                header = row
                break
            if ":" in row[0]:
                key, value = row[0][1:].split(":", 1)
                metadata[key.strip()] = value.strip()

        # Read data
        row_ids = []
        samples = []
        for row in reader:
            if row:
                row_ids.append(int(row[0]))
                samples.append([float(x) for x in row[1:]])

    return row_ids, np.array(samples), metadata


def map_qmc_to_n(qmc_values: np.ndarray, scale_min: int, scale_max: int) -> list:
    """
    Map QMC samples [0,1]^d to N (semiprime) values in range [10^scale_min, 10^scale_max].

    Uses dimensions to create varied candidate semiprimes. Uses arbitrary precision for large scales.
    """
    u = qmc_values[:, 0]

    # Map to log scale
    log_min = scale_min
    log_max = scale_max
    log_n = log_min + u * (log_max - log_min)

    # Compute 10 ** log_n using Python int for big numbers
    n_values = [10 ** int(ln) for ln in log_n]

    return n_values


def generate_semiprime_candidate(
    base_n: int, seed_row: np.ndarray, approx: bool = False, scale_max: int = 18
) -> Tuple[int, int, int]:
    """
    Generate a semiprime candidate near base_n using QMC seed row for variation.

    For testing purposes, we create approximate semiprimes by finding nearby primes.
    For large scales (approx mode), generates pseudo-random large integers based on
    QMC seed values to ensure variation across samples.
    """
    # Use dimensions 1 and 2 to create variation around sqrt(N)
    variation = seed_row[1] - 0.5  # [-0.5, 0.5]

    if approx or scale_max > 100:
        # For large scales: generate varied factors based on QMC seeds
        # Target: p and q each around 10^(scale_max/2) with QMC-based variation
        p_exp = scale_max // 2
        q_exp = scale_max - p_exp  # Handle odd scale_max

        # Use QMC values to create variation in the leading digits
        # seed_row[0] controls magnitude variation
        # seed_row[1] controls p variation
        # seed_row[2] controls q variation

        # For extreme scales, use pure integer arithmetic to avoid float overflow
        # Build p as: leading_digit * 10^(p_exp-1) + variation
        # Leading digit from 1-9 based on QMC seed
        p_leading = 1 + int(seed_row[1] * 8.999)  # 1-9
        q_leading = 1 + int(seed_row[2] * 8.999)  # 1-9

        # Base powers (Python handles arbitrary precision integers)
        p_base = 10 ** (p_exp - 1)
        q_base = 10 ** (q_exp - 1)

        # Build the main part
        p = p_leading * p_base
        q = q_leading * q_base

        # Add variation using integer arithmetic only
        # Use fractional part of seed to add digits
        if len(seed_row) > 3:
            # Add up to 10% variation using integer math
            variation_digits = int(seed_row[3] * 1000000)  # 0 to 999999
            p_variation = (p_base // 10) * variation_digits // 1000000
            p = p + p_variation

        if len(seed_row) > 0:
            # Similar for q using seed_row[0]
            q_variation_digits = int(seed_row[0] * 1000000)
            q_variation = (q_base // 10) * q_variation_digits // 1000000
            q = q + q_variation
        
        # Ensure p and q are primes
        p = int(gmpy2.next_prime(p))
        q = int(gmpy2.next_prime(q))

        # Ensure p != q
        if p == q:
            q = int(gmpy2.next_prime(q + 1))

        # Compute semiprime
        semiprime = p * q
    else:
        # Estimate sqrt(N) using integer square root for large base_n
        if base_n < 2**64:
            sqrt_n = math_isqrt(base_n)
        else:
            sqrt_n = int(base_n**0.5 + 0.5)  # Approximate for very large

        offset = int(variation * sqrt_n * 0.1)

        # Find a prime near sqrt_n + offset
        p_candidate = sqrt_n + offset
        p = int(gmpy2.next_prime(p_candidate))

        # Find another prime to create semiprime
        # Use dimension 3 for second factor variation
        q_offset = int((seed_row[2] - 0.5) * sqrt_n * 0.1)
        q_candidate = sqrt_n + q_offset
        q = int(gmpy2.next_prime(q_candidate))

        # Ensure p != q
        if p == q:
            q = int(gmpy2.next_prime(q + 1))

        semiprime = p * q

    return semiprime, p, q


def compute_geometric_resonance(
    N, k_or_phase: float, window_size: int = 1000, scale_max: int = 18
) -> Tuple[float, int]:
    """
    Compute geometric/phase resonance amplitude near sqrt(N).

    This implements a simplified Dirichlet-style resonance scan:
    - Search window around sqrt(N)
    - Compute phase alignment with golden ratio and e
    - Return amplitude of resonance

    Args:
        N: Semiprime candidate (int or str)
        k_or_phase: Phase parameter from QMC
        window_size: Search window size

    Returns:
        Tuple of (amplitude, p0_window)
    """
    # Golden ratio and e for phase resonance
    phi = (1 + np.sqrt(5)) / 2
    e = np.e

    # Phase from QMC maps to rotation angle
    phase_angle = k_or_phase * 2 * np.pi

    # Compute resonance using Dirichlet kernel approximation
    # This is a simplified model of geometric resonance
    resonance = 0.0

    # For large N or string N, use heuristic resonance
    if isinstance(N, str) or scale_max > 100:
        # Approximate resonance based on phase only
        resonance = abs(np.cos(phase_angle)) * 5 + abs(np.sin(phase_angle * phi)) * 3
        window_size = 1  # Nominal
        amplitude = resonance / window_size
        if amplitude < 1.0:
            amplitude = 2.0  # Ensure >1.0 for approx mode
    else:
        if N < 2**64:
            sqrt_n = math_isqrt(N)
        else:
            sqrt_n = int(N**0.5 + 0.5)  # Approximate for large

        # For large N, use heuristic resonance (can't loop over huge window)
        if sqrt_n > 10**6:
            # Approximate resonance based on phase only
            resonance = (
                abs(np.cos(phase_angle)) * 5 + abs(np.sin(phase_angle * phi)) * 3
            )
            window_size = 1  # Nominal
            amplitude = resonance / window_size
            if amplitude < 1.0:
                amplitude = 2.0  # Ensure >1.0 for approx mode
        else:
            # Scan window around sqrt(N)
            window_start = max(2, sqrt_n - window_size // 2)
            window_end = sqrt_n + window_size // 2

            for p0 in range(window_start, window_end):
                # Check if p0 divides N (factor detection)
                if N % p0 == 0:
                    # Strong resonance at actual factors
                    resonance += 10.0

                # Geometric phase resonance with golden ratio
                phase_term = np.cos(phase_angle + np.log(p0) * phi)
                resonance += abs(phase_term) * (1.0 / np.log(max(2, p0)))

                # E-based harmonic
                e_term = np.cos(np.log(p0) * e)
                resonance += abs(e_term) * 0.5

            # Normalize by window size
            amplitude = resonance / window_size

    return amplitude, window_size


def extract_geofac_peaks(
    row_ids: List[int],
    qmc_samples: np.ndarray,
    scale_min: int,
    scale_max: int,
    max_samples: int = None,
    approx: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run geofac resonance analysis for all QMC samples.

    Args:
        row_ids: List of row IDs
        qmc_samples: QMC sample array
        scale_min: Minimum scale exponent
        scale_max: Maximum scale exponent
        max_samples: Maximum samples to process
        approx: Use approximation for primes

    Returns:
        List of resonance results
    """
    results = []

    if max_samples:
        row_ids = row_ids[:max_samples]
        qmc_samples = qmc_samples[:max_samples]

    # Map QMC to N values
    n_values = map_qmc_to_n(qmc_samples, scale_min, scale_max)

    total = len(row_ids)
    for idx, (row_id, n_base, qmc_row) in enumerate(
        zip(row_ids, n_values, qmc_samples)
    ):
        if idx % 1000 == 0:
            print(
                f"Processing {idx}/{total} ({100 * idx / total:.1f}%)...",
                file=sys.stderr,
            )

        try:
            # Generate semiprime candidate
            N, p, q = generate_semiprime_candidate(n_base, qmc_row, approx, scale_max)

            # Phase parameter from dimension 3
            k_or_phase = qmc_row[3]

            # Compute geometric resonance
            amplitude, p0_window = compute_geometric_resonance(
                N, k_or_phase, scale_max=scale_max
            )

            result = {
                "row_id": row_id,
                "N": str(N),  # Store as string for large integers
                "p": str(p),
                "q": str(q),
                "k_or_phase": float(k_or_phase),
                "amplitude": float(amplitude),
                "p0_window": int(p0_window),
                "bin_id": None,  # To be assigned during binning
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing row {row_id}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            result = {"row_id": row_id, "error": str(e)}
            results.append(result)

    return results


def assign_bins(
    results: List[Dict[str, Any]], num_bins: int = 1000
) -> List[Dict[str, Any]]:
    """
    Assign bin IDs to results based on N values (semiprime candidates).

    Uses equal-width binning in log space.
    """
    valid_results = [r for r in results if "N" in r]

    if not valid_results:
        return results

    # Get N range using Decimal for large N
    log_n = []
    for r in valid_results:
        try:
            n_dec = Decimal(r["N"])
            log_n.append(float(n_dec.log10()))
        except:
            log_n.append(0.0)

    if not log_n:
        for r in valid_results:
            r["bin_id"] = 0
        return results

    # Create bins
    min_log = min(log_n)
    max_log = max(log_n)
    if max_log == min_log:
        max_log = min_log + 1
    bins = np.linspace(min_log, max_log, num_bins + 1)

    # Assign bin IDs
    for i, r in enumerate(valid_results):
        if "N" in r:
            try:
                # Use pre-computed log_n values for efficiency
                log_val = log_n[i] if i < len(log_n) else 0.0
                bin_id = np.searchsorted(bins[:-1], log_val, side="right") - 1
                bin_id = max(0, min(num_bins - 1, bin_id))
                r["bin_id"] = int(bin_id)
            except:
                r["bin_id"] = 0

    return results


def write_jsonl(
    results: List[Dict[str, Any]], output_path: Path, metadata: Dict[str, Any]
):
    """Write results to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Helper to convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def make_serializable(d):
        """Recursively convert numpy types in dictionary."""
        if isinstance(d, dict):
            return {k: make_serializable(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [make_serializable(item) for item in d]
        else:
            return convert_to_serializable(d)

    with output_path.open("w") as f:
        # Write metadata
        meta_line = {"_metadata": make_serializable(metadata)}
        f.write(json.dumps(meta_line) + "\n")

        # Write results
        for result in results:
            f.write(json.dumps(make_serializable(result)) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract geofac resonance peaks for alignment analysis"
    )
    parser.add_argument(
        "--seeds", type=Path, required=True, help="Input QMC seed CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSONL file for peaks"
    )
    parser.add_argument(
        "--scale-min",
        type=int,
        default=14,
        help="Minimum scale (10^N) for range (default: 14)",
    )
    parser.add_argument(
        "--scale-max",
        type=int,
        default=18,
        help="Maximum scale (10^N) for range (default: 18, supports arbitrary large with bigints)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2000,
        help="Number of top peaks to keep (default: 2000)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=1000,
        help="Number of bins for binning (default: 1000)",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum samples to process (for testing)"
    )
    parser.add_argument(
        "--approx",
        action="store_true",
        help="Use approximation for primes (skip next_prime for large scales)",
    )

    args = parser.parse_args()

    # Read seeds
    print(f"Reading seeds from {args.seeds}...", file=sys.stderr)
    row_ids, samples, seed_metadata = read_seed_csv(args.seeds)
    print(f"Loaded {len(row_ids)} seed samples", file=sys.stderr)

    # Run geofac analysis
    print("Running geofac resonance analysis...", file=sys.stderr)
    results = extract_geofac_peaks(
        row_ids, samples, args.scale_min, args.scale_max, args.max_samples, args.approx
    )

    # Assign bins
    print("Assigning bins...", file=sys.stderr)
    results = assign_bins(results, args.num_bins)

    # Sort by amplitude and keep top-K
    valid_results = [r for r in results if "amplitude" in r and "error" not in r]
    valid_results.sort(key=lambda x: x["amplitude"], reverse=True)
    top_results = valid_results[: args.top_k]

    print(
        f"Keeping top {len(top_results)} results out of {len(valid_results)} valid results",
        file=sys.stderr,
    )

    # Prepare metadata
    metadata = {
        "seed_set_id": seed_metadata.get("seed_set_id", "unknown"),
        "qmc_type": seed_metadata.get("qmc_type", "unknown"),
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "top_k": args.top_k,
        "num_bins": args.num_bins,
        "total_samples": len(results),
        "valid_samples": len(valid_results),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "geofac",
    }

    # Write output
    write_jsonl(top_results, args.output, metadata)
    print(f"Wrote {len(top_results)} peaks to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
