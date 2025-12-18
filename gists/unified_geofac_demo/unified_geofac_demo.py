#!/usr/bin/env python3
"""
Unified GeoFac Demo: Blind Factorization Combining Balanced + Adaptive Engines

This script demonstrates factorization of semiprimes using a unified approach:
1. Balanced GeoFac: Resonance scanning near √N
2. Adaptive window: Z5D-scored candidate generation and testing

Usage: python unified_geofac_demo.py N
where N is the semiprime to factor (no access to p,q assumed).

Requirements: numpy, scipy, gmpy2, mpmath
"""

import sys
import time
import random
import argparse
import numpy as np
import gmpy2
import mpmath


# Z5D Adapter functions (ported from z5d_adapter.py)
def z5d_n_est(p_str: str) -> gmpy2.mpz:
    p = gmpy2.mpz(p_str)
    if p < 2:
        return gmpy2.mpz(1)

    bits = int(gmpy2.bit_length(p))
    mpmath.mp.dps = max(100, int(bits * 0.4) + 200)

    p_mpf = mpmath.mpf(str(p))
    ln_p = mpmath.log(p_mpf)

    if ln_p <= 0:
        return gmpy2.mpz(1)

    inv_ln_p = mpmath.mpf(1) / ln_p
    n_est = (p_mpf / ln_p) * (1 + inv_ln_p + 2 * inv_ln_p * inv_ln_p)

    n_est_str = mpmath.nstr(n_est, int(mpmath.mp.dps), strip_zeros=False).split(".")[0]
    return gmpy2.mpz(n_est_str) if n_est_str and n_est_str != "-" else gmpy2.mpz(1)


def compute_z5d_score(p_str: str, n_est: gmpy2.mpz) -> float:
    p = gmpy2.mpz(p_str)
    if p <= 0:
        return 0.0

    predicted_p = z5d_predict_nth_prime(n_est)
    diff = abs(p - predicted_p)

    if diff == 0:
        return -100.0

    bits = max(int(gmpy2.bit_length(p)), int(gmpy2.bit_length(diff)))
    mpmath.mp.dps = max(100, int(bits * 0.4) + 200)

    diff_mpf = mpmath.mpf(str(diff))
    p_mpf = mpmath.mpf(str(p))
    rel_error = diff_mpf / p_mpf
    log_rel_error = mpmath.log10(rel_error)

    try:
        score = float(log_rel_error)
        if score > 1e10:
            score = 1e10
        elif score < -1e10:
            score = -1e10
        return score
    except (OverflowError, ValueError):
        diff_bits = int(gmpy2.bit_length(diff))
        p_bits = int(gmpy2.bit_length(p))
        return (diff_bits - p_bits) * 0.301


def z5d_predict_nth_prime(n: gmpy2.mpz) -> gmpy2.mpz:
    if n <= 0:
        return gmpy2.mpz(2)
    if n == 1:
        return gmpy2.mpz(2)
    if n == 2:
        return gmpy2.mpz(3)
    if n == 3:
        return gmpy2.mpz(5)
    if n == 4:
        return gmpy2.mpz(7)
    if n == 5:
        return gmpy2.mpz(11)

    bits = int(gmpy2.bit_length(n))
    mpmath.mp.dps = max(100, int(bits * 0.4) + 200)

    n_mpf = mpmath.mpf(str(n))
    ln_n = mpmath.log(n_mpf)

    if ln_n <= 0:
        return gmpy2.mpz(2)

    ln_ln_n = mpmath.log(ln_n)
    correction = (ln_ln_n - 2) / ln_n
    predicted = n_mpf * (ln_n + ln_ln_n - 1 + correction)

    pred_str = mpmath.nstr(predicted, int(mpmath.mp.dps), strip_zeros=False).split(".")[
        0
    ]
    return gmpy2.mpz(pred_str) if pred_str and pred_str != "-" else gmpy2.mpz(2)


# Balanced GeoFac resonance (ported from run_geofac_peaks_mod.py)
def compute_geometric_resonance(
    N, k_or_phase: float, window_size: int = 1000, scale_max: int = 18
) -> tuple[float, int]:
    phi = (1 + np.sqrt(5)) / 2
    e_const = np.e
    phase_angle = k_or_phase * 2 * np.pi
    resonance = 0.0

    if isinstance(N, str) or scale_max > 100:
        resonance = abs(np.cos(phase_angle)) * 5 + abs(np.sin(phase_angle * phi)) * 3
        window_size = 1
        amplitude = resonance / window_size
        if amplitude < 1.0:
            amplitude = 2.0
    else:
        if N < 2**64:
            sqrt_n = int(np.sqrt(N) + 0.5)
        else:
            sqrt_n = int(N**0.5 + 0.5)

        if sqrt_n > 10**6:
            resonance = (
                abs(np.cos(phase_angle)) * 5 + abs(np.sin(phase_angle * phi)) * 3
            )
            window_size = 1
            amplitude = resonance / window_size
            if amplitude < 1.0:
                amplitude = 2.0
        else:
            window_start = max(2, sqrt_n - window_size // 2)
            window_end = sqrt_n + window_size // 2

            for p0 in range(window_start, window_end):
                if N % p0 == 0:
                    resonance += 10.0
                phase_term = np.cos(phase_angle + np.log(p0) * phi)
                resonance += abs(phase_term) * (1.0 / np.log(max(2, p0)))
                e_term = np.cos(np.log(p0) * e_const)
                resonance += abs(e_term) * 0.5

            amplitude = resonance / window_size

    return amplitude, window_size


# Unified demo function
def unified_geofac_demo(
    N_str: str, verbose: bool = False, log_csv: str = None
) -> tuple[int | None, int | None, dict]:
    N = gmpy2.mpz(N_str)
    if N % 2 == 0:
        return (
            2,
            int(N // 2),
            {
                "balanced_candidates": 0,
                "balanced_time": 0.0,
                "adaptive_candidates": 0,
                "adaptive_time": 0.0,
                "best_score": None,
            },
        )

    sqrt_N = gmpy2.isqrt(N)
    int_sqrt = int(sqrt_N)

    # Phase 1: Balanced GeoFac - scan window for divisibility
    bit_length = N.bit_length()
    window_size = min(10000, max(1000, bit_length * 10))
    window_start = max(2, int_sqrt - window_size // 2)
    window_end = int_sqrt + window_size // 2

    balanced_candidates = 0
    balanced_start = time.time()
    for p0 in range(window_start, window_end):
        balanced_candidates += 1
        if N % p0 == 0:
            q0 = int(N // p0)
            balanced_time = time.time() - balanced_start
            return (
                min(p0, q0),
                max(p0, q0),
                {
                    "balanced_candidates": balanced_candidates,
                    "balanced_time": balanced_time,
                    "adaptive_candidates": 0,
                    "adaptive_time": 0.0,
                    "best_score": None,
                },
            )
    balanced_time = time.time() - balanced_start

    # Phase 2: Adaptive window with iterative expansion
    adaptive_start = time.time()
    window_schedule = [13, 20, 30, 50, 75, 100, 150, 200, 300]  # percentages
    best_overall_score = None
    total_candidates_tested = 0
    windows_exhausted = 0

    for window_pct in window_schedule:
        window_radius = (sqrt_N * window_pct) // 100
        search_min = sqrt_N - window_radius
        search_max = sqrt_N + window_radius

        # Generate candidates for THIS window
        # Match validation experiment density: 1M for ±13% window, 100K for others
        if window_pct == 13:
            num_candidates = 1000000  # 1M for ±13% window
        else:
            num_candidates = 100000  # 100K for wider windows

        # DIAGNOSTIC: Log window bounds
        if verbose:
            print(f"\n[Window ±{window_pct}%]", file=sys.stderr)
            print(f"  Range: [{int(search_min)}, {int(search_max)}]", file=sys.stderr)
            print(f"  Width: {int(search_max - search_min)}", file=sys.stderr)
            print(f"  Generating {num_candidates:,} candidates...", file=sys.stderr)

        # Calculate odd-only bounds for uniform sampling over odds
        search_min_odd = search_min if search_min % 2 == 1 else search_min + 1
        search_max_odd = search_max if search_max % 2 == 1 else search_max - 1
        total_odds = ((search_max_odd - search_min_odd) // 2) + 1

        if verbose:
            print(f"  Odd candidates in window: {total_odds:,}", file=sys.stderr)
            print(
                f"  Sampling ratio: {num_candidates}/{total_odds} = {100 * num_candidates / total_odds:.4f}%",
                file=sys.stderr,
            )

        candidates = []
        random.seed(127 + window_pct)

        for _ in range(num_candidates):
            k = random.randrange(total_odds)
            candidate = search_min_odd + (2 * k)
            candidates.append(gmpy2.mpz(candidate))

        # DIAGNOSTIC: Check if true factors in candidate set (for N_127)
        if verbose and N == gmpy2.mpz("137524771864208156028430259349934309717"):
            P_127 = gmpy2.mpz("10508623501177419659")
            Q_127 = gmpy2.mpz("13086849276577416863")

            p_in_window = search_min <= P_127 <= search_max
            q_in_window = search_min <= Q_127 <= search_max
            p_sampled = P_127 in candidates
            q_sampled = Q_127 in candidates

            print(f"  True factors:", file=sys.stderr)
            print(
                f"    p={int(P_127)}: in_window={p_in_window}, sampled={p_sampled}",
                file=sys.stderr,
            )
            print(
                f"    q={int(Q_127)}: in_window={q_in_window}, sampled={q_sampled}",
                file=sys.stderr,
            )

        # Score candidates
        scored = []
        for c in candidates:
            n_est = z5d_n_est(str(c))
            score = compute_z5d_score(str(c), n_est)
            scored.append((score, int(c)))

    scored.sort(key=lambda x: x[0])
    total_candidates_tested += len(scored)

    # Track best score
    if scored and (best_overall_score is None or scored[0][0] < best_overall_score):
        best_overall_score = scored[0][0]

    # DIAGNOSTIC: Log score statistics
    if verbose:
        print(
            f"  Score range: [{scored[0][0]:.4f}, {scored[-1][0]:.4f}]", file=sys.stderr
        )
        print(
            f"  Testing top {min(10000, len(scored))} for divisibility...",
            file=sys.stderr,
        )

        # Check where true factors rank (for N_127)
        if N == gmpy2.mpz("137524771864208156028430259349934309717"):
            P_127 = gmpy2.mpz("10508623501177419659")
            Q_127 = gmpy2.mpz("13086849276577416863")
            if P_127 in candidates or Q_127 in candidates:
                for rank, (score, c) in enumerate(scored):
                    if c == int(P_127):
                        print(
                            f"    ⚠ p ranked #{rank + 1} with score {score:.4f}",
                            file=sys.stderr,
                        )
                    if c == int(Q_127):
                        print(
                            f"    ⚠ q ranked #{rank + 1} with score {score:.4f}",
                            file=sys.stderr,
                        )

        # Test top 10000 for divisibility, with local search around each (Z5D enriches near factors)
        for score, c in scored[:10000]:
            # Check the candidate itself
            if N % c == 0:
                q = int(N // c)
                adaptive_time = time.time() - adaptive_start
                return (
                    min(c, q),
                    max(c, q),
                    {
                        "balanced_candidates": balanced_candidates,
                        "balanced_time": balanced_time,
                        "adaptive_candidates": total_candidates_tested,
                        "adaptive_time": adaptive_time,
                        "best_score": best_overall_score,
                        "window_used": f"{window_pct}%",
                        "windows_exhausted": windows_exhausted + 1,
                    },
                )
            # Local search: check ±20 nearby odds for divisibility
            for offset in range(-20, 21):
                if offset == 0:
                    continue
                nearby = c + 2 * offset  # stay on odds
                if nearby < search_min or nearby > search_max:
                    continue
                if N % nearby == 0:
                    q = int(N // nearby)
                    adaptive_time = time.time() - adaptive_start
                    return (
                        min(nearby, q),
                        max(nearby, q),
                        {
                            "balanced_candidates": balanced_candidates,
                            "balanced_time": balanced_time,
                            "adaptive_candidates": total_candidates_tested,
                            "adaptive_time": adaptive_time,
                            "best_score": best_overall_score,
                            "window_used": f"{window_pct}%",
                            "windows_exhausted": windows_exhausted + 1,
                        },
                    )

        windows_exhausted += 1

    # All windows exhausted without finding factor
    adaptive_time = time.time() - adaptive_start
    return (
        None,
        None,
        {
            "balanced_candidates": balanced_candidates,
            "balanced_time": balanced_time,
            "adaptive_candidates": total_candidates_tested,
            "adaptive_time": adaptive_time,
            "best_score": best_overall_score,
            "windows_exhausted": windows_exhausted,
        },
    )


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified GeoFac Demo: Blind semiprime factorization"
    )
    parser.add_argument("N", type=str, help="Semiprime to factor")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose diagnostic logging"
    )
    parser.add_argument(
        "--log-csv", type=str, help="Save per-window candidate stats to CSV"
    )
    args = parser.parse_args()

    N_str = args.N
    try:
        int(N_str)
    except ValueError:
        print("Error: N must be an integer")
        sys.exit(1)

    start_time = time.time()
    p, q, metadata = unified_geofac_demo(
        N_str, verbose=args.verbose, log_csv=args.log_csv
    )
    end_time = time.time()

    if p is not None and q is not None:
        print("Success")
        print(f"Factor pair: {p}, {q}")
        print(f"Verification: {p} * {q} = {p * q}")
        print(f"Balanced phase: {metadata['balanced_candidates']} candidates scanned in {metadata['balanced_time']:.4f}s")
        print(f"Adaptive phase: {metadata['adaptive_candidates']} total candidates across {metadata.get('window_used', 'all')} windows in {metadata['adaptive_time']:.4f}s")
        if metadata['best_score'] is not None:
            print(f"Best Z5D score: {metadata['best_score']:.4f}")
    else:
        print("Failure")
        print("No factor found within limits")
        print(f"\nDiagnostics:")
        print(f"  N = {N_str}")
        print(f"  √N ≈ {int(gmpy2.isqrt(gmpy2.mpz(N_str)))}")
        print(f"  Total candidates tested: {metadata['adaptive_candidates']:,}")
        print(f"  Best Z5D score: {metadata['best_score']:.4f}")
        print(f"  Windows searched: {metadata.get('windows_exhausted', 0)}")
        print(f"\nTo debug, run with: --verbose --log-csv debug.csv")
    else:
        print("Failure")
        print("No factor found within limits")
        print(
            f"Balanced phase: {metadata['balanced_candidates']} candidates scanned in {metadata['balanced_time']:.4f}s"
        )
        print(
            f"Adaptive phase: {metadata['adaptive_candidates']} total candidates across {metadata.get('windows_exhausted', 0)} windows in {metadata['adaptive_time']:.4f}s"
        )
        if metadata["best_score"] is not None:
            print(f"Best Z5D score: {metadata['best_score']:.4f}")
