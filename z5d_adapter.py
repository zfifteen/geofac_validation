#!/usr/bin/env python3
"""
Z5D Adapter: Compute Z5D scores and n_est for Geofac peaks.
Handles ARBITRARY PRECISION numbers with gmpy2 and mpmath.
NO uint64_t, NO fixed-width integers - scales to 10^1233 and beyond.

Implements the Z5D nth-prime predictor using PNT approximation:
  p(n) ≈ n * (ln(n) + ln(ln(n)) - 1 + (ln(ln(n)) - 2) / ln(n))
"""

import sys
import json
import gmpy2
import mpmath


def z5d_n_est(p_str: str) -> gmpy2.mpz:
    """
    Estimate prime index n such that p(n) ≈ p.
    Uses asymptotic PNT approximation: pi(x) ≈ x / ln(x) * (1 + 1/ln(x) + 2/ln(x)^2)

    ALL arithmetic uses arbitrary precision - no float conversion that could overflow.
    """
    p = gmpy2.mpz(p_str)
    if p < 2:
        return gmpy2.mpz(1)

    # Set mpmath precision based on bit length of p
    # Need enough precision to handle numbers up to 10^1233 (~4100 bits)
    bits = int(gmpy2.bit_length(p))
    # Use generous precision: ~0.4 decimal digits per bit + buffer
    mpmath.mp.dps = max(100, int(bits * 0.4) + 200)

    # Convert to mpmath mpf (arbitrary precision float) - use string to avoid any float intermediates
    p_mpf = mpmath.mpf(str(p))

    # Compute ln(p) in arbitrary precision
    ln_p = mpmath.log(p_mpf)

    if ln_p <= 0:
        return gmpy2.mpz(1)

    # Prime counting function approximation: pi(x) ≈ x / ln(x) * (1 + 1/ln(x) + 2/ln(x)^2)
    # This is the asymptotic expansion of li(x) which works for ALL scales
    inv_ln_p = mpmath.mpf(1) / ln_p
    n_est = (p_mpf / ln_p) * (1 + inv_ln_p + 2 * inv_ln_p * inv_ln_p)

    # Convert back to gmpy2.mpz via string (no float intermediate!)
    n_est_str = mpmath.nstr(n_est, int(mpmath.mp.dps), strip_zeros=False).split('.')[0]
    return gmpy2.mpz(n_est_str) if n_est_str and n_est_str != '-' else gmpy2.mpz(1)


def z5d_predict_nth_prime(n: gmpy2.mpz) -> gmpy2.mpz:
    """
    Predict the nth prime using PNT-based approximation.

    Formula: p(n) ≈ n * (ln(n) + ln(ln(n)) - 1 + (ln(ln(n)) - 2) / ln(n))

    ALL arithmetic uses arbitrary precision - scales to ANY n.
    """
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

    # Set precision based on magnitude of n
    bits = int(gmpy2.bit_length(n))
    mpmath.mp.dps = max(100, int(bits * 0.4) + 200)

    # Convert to mpmath via string - NEVER use int() for huge numbers going to float
    n_mpf = mpmath.mpf(str(n))
    ln_n = mpmath.log(n_mpf)

    if ln_n <= 0:
        return gmpy2.mpz(2)

    ln_ln_n = mpmath.log(ln_n)

    # PNT approximation with correction terms
    # p(n) ≈ n * (ln(n) + ln(ln(n)) - 1 + (ln(ln(n)) - 2) / ln(n))
    correction = (ln_ln_n - 2) / ln_n
    predicted = n_mpf * (ln_n + ln_ln_n - 1 + correction)

    # Convert back to gmpy2.mpz via string (no float intermediate!)
    pred_str = mpmath.nstr(predicted, int(mpmath.mp.dps), strip_zeros=False).split('.')[0]
    return gmpy2.mpz(pred_str) if pred_str and pred_str != '-' else gmpy2.mpz(2)


def compute_z5d_score(p_str: str, n_est: gmpy2.mpz) -> float:
    """
    Compute Z5D score as normalized log-relative deviation.

    Score = log10(|p - predicted_p| / p) scaled to be comparable across all magnitudes.
    Lower (more negative) scores indicate better alignment with Z5D model.

    For valid predictions, score typically ranges from -10 (excellent) to +10 (poor).
    """
    p = gmpy2.mpz(p_str)
    if p <= 0:
        return 0.0

    predicted_p = z5d_predict_nth_prime(n_est)

    # Compute absolute difference
    diff = abs(p - predicted_p)

    if diff == 0:
        return -100.0  # Perfect prediction (unlikely but handle it)

    # Set precision for mpmath based on larger of p or diff
    bits = max(int(gmpy2.bit_length(p)), int(gmpy2.bit_length(diff)))
    mpmath.mp.dps = max(100, int(bits * 0.4) + 200)

    # Compute log-relative error: log10(|p - p'| / p)
    # This normalizes across all scales
    diff_mpf = mpmath.mpf(str(diff))
    p_mpf = mpmath.mpf(str(p))

    # Relative error
    rel_error = diff_mpf / p_mpf

    # Log10 of relative error - this is always a reasonable-sized number
    log_rel_error = mpmath.log10(rel_error)

    # Convert to float - this should always work since log values are small
    try:
        score = float(log_rel_error)
        # Clamp to reasonable range for JSON
        if score > 1e10:
            score = 1e10
        elif score < -1e10:
            score = -1e10
        return score
    except (OverflowError, ValueError):
        # Fallback: estimate from bit lengths
        diff_bits = int(gmpy2.bit_length(diff))
        p_bits = int(gmpy2.bit_length(p))
        return (diff_bits - p_bits) * 0.301  # log10(2) ≈ 0.301


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if '"_metadata"' in line:
            print(line)
            continue

        # Parse JSON
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        p_str = data.get("p", "")
        q_str = data.get("q", "")

        if not p_str or not q_str:
            print(line)
            continue

        # Compute n_est for both factors - ALL ARBITRARY PRECISION
        try:
            n_est_p = z5d_n_est(p_str)
            n_est_q = z5d_n_est(q_str)

            # Compute actual Z5D scores (deviation from prediction)
            score_p = compute_z5d_score(p_str, n_est_p)
            score_q = compute_z5d_score(q_str, n_est_q)
        except Exception as e:
            # Log error to stderr, return error indicators
            print(f"Z5D Error: {e}", file=sys.stderr)
            n_est_p = gmpy2.mpz(0)
            n_est_q = gmpy2.mpz(0)
            score_p = -1.0
            score_q = -1.0

        # Output enriched JSON
        enriched = data.copy()
        enriched["z5d_score_p"] = score_p
        enriched["z5d_n_est_p"] = str(n_est_p)
        enriched["z5d_score_q"] = score_q
        enriched["z5d_n_est_q"] = str(n_est_q)

        print(json.dumps(enriched))


if __name__ == "__main__":
    main()
