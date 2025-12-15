#!/usr/bin/env python3
"""
Generate QMC (Quasi-Monte Carlo) seed sequences for reproducible experiments.

Supports Sobol and Halton sequences with configurable parameters.
Outputs seeds to CSV format with row indices for exact reproducibility.

Usage:
    python generate_qmc_seeds.py --type sobol --samples 1000 --output ../artifacts/seedsets/phi_qmc_001.csv
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import qmc


def generate_sobol_sequence(
    num_samples: int, dimensions: int = 5, seed: int = None
) -> np.ndarray:
    """
    Generate Sobol sequence for quasi-Monte Carlo sampling.

    Args:
        num_samples: Number of samples to generate
        dimensions: Number of dimensions (default 5 for Z5D)
        seed: Random seed for reproducibility

    Returns:
        Array of shape (num_samples, dimensions) with values in [0, 1)
    """
    sampler = qmc.Sobol(d=dimensions, scramble=True, seed=seed)
    # Generate samples (Sobol sequences require power of 2, but we take what we need)
    samples = sampler.random(n=num_samples)
    return samples


def generate_halton_sequence(
    num_samples: int, dimensions: int = 5, seed: int = None
) -> np.ndarray:
    """
    Generate Halton sequence for quasi-Monte Carlo sampling.

    Args:
        num_samples: Number of samples to generate
        dimensions: Number of dimensions (default 5 for Z5D)
        seed: Random seed for reproducibility

    Returns:
        Array of shape (num_samples, dimensions) with values in [0, 1)
    """
    sampler = qmc.Halton(d=dimensions, scramble=True, seed=seed)
    samples = sampler.random(n=num_samples)
    return samples


def write_seed_csv(samples: np.ndarray, output_path: Path, metadata: dict):
    """
    Write QMC samples to CSV with metadata header.

    Args:
        samples: Array of QMC samples
        output_path: Path to output CSV file
        metadata: Dictionary of metadata to include in header
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Write metadata as comments
        writer.writerow(["# QMC Seed Set"])
        for key, value in metadata.items():
            writer.writerow([f"# {key}: {value}"])
        writer.writerow(["#"])

        # Write header
        num_dims = samples.shape[1]
        header = ["row_id"] + [f"dim_{i}" for i in range(num_dims)]
        writer.writerow(header)

        # Write data
        for idx, sample in enumerate(samples):
            row = [idx] + [f"{val:.16e}" for val in sample]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Generate QMC seed sequences for Z5D-Geofac alignment experiments"
    )
    parser.add_argument(
        "--type",
        choices=["sobol", "halton"],
        default="sobol",
        help="Type of QMC sequence (default: sobol)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200000,
        help="Number of samples to generate (default: 200000)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=5,
        help="Number of dimensions (default: 5 for Z5D)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--set-id",
        default="phi_qmc_001",
        help="Seed set identifier (default: phi_qmc_001)",
    )

    args = parser.parse_args()

    print(f"Generating {args.type} sequence with {args.samples} samples...")

    # Generate samples
    if args.type == "sobol":
        samples = generate_sobol_sequence(args.samples, args.dimensions, args.seed)
    else:
        samples = generate_halton_sequence(args.samples, args.dimensions, args.seed)

    # Prepare metadata
    metadata = {
        "seed_set_id": args.set_id,
        "qmc_type": args.type,
        "num_samples": args.samples,
        "dimensions": args.dimensions,
        "seed": args.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scipy.stats.qmc",
        "scramble": "true",
    }

    # Write to CSV
    write_seed_csv(samples, args.output, metadata)
    print(f"Wrote {args.samples} samples to {args.output}")
    print(f"Seed set ID: {args.set_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
