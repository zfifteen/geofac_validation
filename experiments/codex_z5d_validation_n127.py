#!/usr/bin/env python3
"""
Validate whether Z5D alignment scoring enriches candidates near the true factors of N_127.

Experiment (falsifiable):
  1) Uniformly sample odd candidates in the ±13% window around sqrt(N_127).
  2) Score each candidate with z5d_adapter.compute_z5d_score() (lower = better alignment).
  3) Define `z5d_resonance = -z5d_score` (higher = better) and rank candidates by it.
  4) Measure Top-K enrichment near the ground-truth factors p and q versus a uniform baseline.

Outputs:
  - Per-candidate CSV (optional; can be large for 1M rows)
  - Summary JSON (enrichment metrics, baseline expectations, statistical tests)

Example:
  python3 experiments/z5d_validation_n127.py --num-candidates 1000000 \\
    --processes 8 \\
    --out-csv artifacts/z5d_validation_n127.csv \\
    --summary-json artifacts/z5d_validation_n127_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from heapq import heappush, heapreplace
from pathlib import Path
from typing import Iterator, Sequence

import gmpy2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from z5d_adapter import compute_z5d_score, z5d_n_est

try:
    import scipy
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency for tests/analysis
    scipy = None
    scipy_stats = None


N_127 = gmpy2.mpz("137524771864208156028430259349934309717")
P_127 = gmpy2.mpz("10508623501177419659")
Q_127 = gmpy2.mpz("13086849276577416863")


def _is_odd(value: int) -> bool:
    return (value & 1) == 1


def _odd_bounds(min_inclusive: int, max_inclusive: int) -> tuple[int, int]:
    min_odd = min_inclusive if _is_odd(min_inclusive) else min_inclusive + 1
    max_odd = max_inclusive if _is_odd(max_inclusive) else max_inclusive - 1
    if min_odd > max_odd:
        raise ValueError("Empty odd window; check bounds.")
    return min_odd, max_odd


def _count_odds(min_inclusive: int, max_inclusive: int) -> int:
    min_odd, max_odd = _odd_bounds(min_inclusive, max_inclusive)
    return ((max_odd - min_odd) // 2) + 1


def _count_odds_in_intersection(
    window_min: int,
    window_max: int,
    interval_min: int,
    interval_max: int,
) -> int:
    lo = max(window_min, interval_min)
    hi = min(window_max, interval_max)
    if lo > hi:
        return 0
    lo_odd, hi_odd = _odd_bounds(lo, hi)
    if lo_odd > hi_odd:
        return 0
    return ((hi_odd - lo_odd) // 2) + 1


def _relative_window(target: int, pct_numer: int, pct_denom: int = 100) -> tuple[int, int]:
    radius = (target * pct_numer) // pct_denom
    return target - radius, target + radius


def _within_relative(candidate: int, target: int, pct_numer: int, pct_denom: int = 100) -> bool:
    radius = (target * pct_numer) // pct_denom
    return abs(candidate - target) <= radius


def _chunks(values: Sequence[int], chunk_size: int) -> Iterator[list[int]]:
    for i in range(0, len(values), chunk_size):
        yield list(values[i : i + chunk_size])


def _score_chunk(candidates: Sequence[int]) -> list[float]:
    scores: list[float] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        n_est = z5d_n_est(candidate_str)
        scores.append(compute_z5d_score(candidate_str, n_est))
    return scores


def _bootstrap_enrichment(
    flags: Sequence[bool],
    baseline: float,
    *,
    rng: random.Random,
    trials: int,
) -> tuple[float, float] | None:
    if baseline <= 0.0:
        return None
    if not flags or trials <= 0:
        return None

    n = len(flags)
    bits = [1 if f else 0 for f in flags]

    enrichments: list[float] = []
    for _ in range(trials):
        hits = 0
        for _ in range(n):
            hits += bits[rng.randrange(n)]
        enrichments.append((hits / n) / baseline)

    enrichments.sort()
    lo = enrichments[int(0.025 * trials)]
    hi = enrichments[int(0.975 * trials)]
    return lo, hi


@dataclass(frozen=True, order=True)
class TopEntry:
    z5d_resonance: float
    candidate: int
    z5d_score: float
    dist_p: int
    dist_q: int
    dist_sqrt: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Z5D alignment enrichment near N_127 true factors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=100_000,
        help="Number of uniformly sampled odd candidates to score.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=127,
        help="RNG seed for candidate generation and bootstrap reproducibility.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Number of candidates scored per batch (affects progress granularity).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Worker processes for scoring (1 disables multiprocessing).",
    )
    parser.add_argument(
        "--topk-max",
        type=int,
        default=100_000,
        help="Largest Top-K slice retained in-memory for analysis.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("artifacts/z5d_validation_n127.csv"),
        help="Output CSV path (set to '-' to disable CSV writing).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/z5d_validation_n127_summary.json"),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--bootstrap-trials",
        type=int,
        default=200,
        help="Bootstrap trials for enrichment CI (0 disables).",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip KS/Mann-Whitney tests (useful if SciPy unavailable).",
    )
    args = parser.parse_args()

    num_candidates = args.num_candidates
    if num_candidates <= 0:
        raise SystemExit("--num-candidates must be > 0")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    if args.processes <= 0:
        raise SystemExit("--processes must be > 0")

    p_int = int(P_127)
    q_int = int(Q_127)
    sqrt_n = int(gmpy2.isqrt(N_127))

    window_radius = (sqrt_n * 13) // 100
    search_min = sqrt_n - window_radius
    search_max = sqrt_n + window_radius

    win_min_odd, win_max_odd = _odd_bounds(search_min, search_max)
    total_odds = _count_odds(win_min_odd, win_max_odd)

    topk_max = args.topk_max
    if topk_max <= 0:
        raise SystemExit("--topk-max must be > 0")
    topk_max = min(topk_max, num_candidates)

    out_csv_path: Path | None
    if str(args.out_csv) == "-":
        out_csv_path = None
    else:
        out_csv_path = args.out_csv
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Baseline zone definitions (requested in issue #16).
    zones = {
        "within_p_1pct": (p_int, 1),
        "within_q_1pct": (q_int, 1),
        "within_p_or_q_5pct": (None, 5),
    }

    baseline_expected: dict[str, float] = {}
    for name, (target, pct) in zones.items():
        if target is None:
            p_lo, p_hi = _relative_window(p_int, pct)
            q_lo, q_hi = _relative_window(q_int, pct)
            count = _count_odds_in_intersection(win_min_odd, win_max_odd, p_lo, p_hi) + _count_odds_in_intersection(
                win_min_odd, win_max_odd, q_lo, q_hi
            )
        else:
            lo, hi = _relative_window(target, pct)
            count = _count_odds_in_intersection(win_min_odd, win_max_odd, lo, hi)
        baseline_expected[name] = count / total_odds

    baseline_sample_counts = {name: 0 for name in zones}

    top_heap: list[TopEntry] = []

    start = time.time()
    last_report = start

    csv_file = None
    writer: csv.writer | None = None
    if out_csv_path is not None:
        csv_file = out_csv_path.open("w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "row_id",
                "candidate",
                "z5d_score",
                "z5d_resonance",
                "dist_to_p",
                "dist_to_q",
                "dist_to_sqrt",
                "nearest_factor",
                "nearest_factor_dist",
            ]
        )

    def handle_scored_batch(candidates: Sequence[int], scores: Sequence[float]) -> None:
        nonlocal top_heap

        nonlocal baseline_sample_counts
        nonlocal writer
        nonlocal row_id

        for candidate, score in zip(candidates, scores, strict=True):
            resonance = -score
            dist_p = abs(candidate - p_int)
            dist_q = abs(candidate - q_int)
            dist_sqrt = abs(candidate - sqrt_n)
            nearest_factor = "p" if dist_p <= dist_q else "q"
            nearest_dist = dist_p if nearest_factor == "p" else dist_q

            if writer is not None:
                writer.writerow(
                    [
                        row_id,
                        str(candidate),
                        f"{score:.17g}",
                        f"{resonance:.17g}",
                        str(dist_p),
                        str(dist_q),
                        str(dist_sqrt),
                        nearest_factor,
                        str(nearest_dist),
                    ]
                )

            # Baseline counts across the full (uniform) sample.
            if _within_relative(candidate, p_int, 1):
                baseline_sample_counts["within_p_1pct"] += 1
            if _within_relative(candidate, q_int, 1):
                baseline_sample_counts["within_q_1pct"] += 1
            if _within_relative(candidate, p_int, 5) or _within_relative(candidate, q_int, 5):
                baseline_sample_counts["within_p_or_q_5pct"] += 1

            entry = TopEntry(
                z5d_resonance=resonance,
                candidate=candidate,
                z5d_score=score,
                dist_p=dist_p,
                dist_q=dist_q,
                dist_sqrt=dist_sqrt,
            )
            if len(top_heap) < topk_max:
                heappush(top_heap, entry)
            else:
                if entry.z5d_resonance > top_heap[0].z5d_resonance:
                    heapreplace(top_heap, entry)

            row_id += 1

    scored = 0
    remaining = num_candidates
    row_id = 0

    pool = None
    if args.processes > 1:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=args.processes)

    try:
        while remaining > 0:
            batch = min(args.chunk_size, remaining)
            remaining -= batch

            # Uniform over odd integers in [win_min_odd, win_max_odd]:
            # candidate = win_min_odd + 2*k, where k is uniform in [0, total_odds).
            candidates: list[int] = []
            for _ in range(batch):
                k = rng.randrange(total_odds)
                candidates.append(win_min_odd + (2 * k))

            if pool is None:
                scores = _score_chunk(candidates)
                handle_scored_batch(candidates, scores)
            else:
                sub_chunk_size = max(1, len(candidates) // (args.processes * 4))
                parts = list(_chunks(candidates, sub_chunk_size))
                scores_parts = list(pool.map(_score_chunk, parts))
                scores = [s for part in scores_parts for s in part]
                handle_scored_batch(candidates, scores)

            scored += batch

            now = time.time()
            if now - last_report >= 2.0:
                rate = scored / max(1e-9, now - start)
                eta = (num_candidates - scored) / max(1e-9, rate)
                print(
                    f"Scored {scored:,}/{num_candidates:,} candidates "
                    f"({rate:,.0f}/s, ETA {eta:,.1f}s)",
                    file=sys.stderr,
                )
                last_report = now
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

        if csv_file is not None:
            csv_file.close()

    elapsed = time.time() - start

    # Sort Top-K by resonance descending (highest = best alignment).
    top_sorted = sorted(top_heap, key=lambda e: e.z5d_resonance, reverse=True)

    topk_slices = [100, 1_000, 10_000, 100_000]
    topk_slices = [k for k in topk_slices if k <= len(top_sorted)]

    def frac_in_zone(entries: Sequence[TopEntry], zone_name: str) -> float:
        if not entries:
            return 0.0
        if zone_name == "within_p_1pct":
            hits = sum(1 for e in entries if _within_relative(e.candidate, p_int, 1))
        elif zone_name == "within_q_1pct":
            hits = sum(1 for e in entries if _within_relative(e.candidate, q_int, 1))
        elif zone_name == "within_p_or_q_5pct":
            hits = sum(
                1
                for e in entries
                if _within_relative(e.candidate, p_int, 5) or _within_relative(e.candidate, q_int, 5)
            )
        else:
            raise ValueError(f"Unknown zone: {zone_name}")
        return hits / len(entries)

    baseline_sample = {name: baseline_sample_counts[name] / num_candidates for name in zones}

    metrics: dict[str, dict[str, dict[str, float | int | list[float] | None]]] = {}
    bootstrap_rng = random.Random(args.seed + 1)

    for k in topk_slices:
        slice_entries = top_sorted[:k]
        k_metrics: dict[str, dict[str, float | int | list[float] | None]] = {}

        for zone_name in zones:
            top_frac = frac_in_zone(slice_entries, zone_name)
            expected_base = baseline_expected[zone_name]
            sample_base = baseline_sample[zone_name]

            enrich_expected = top_frac / expected_base if expected_base > 0 else math.inf
            enrich_sample = top_frac / sample_base if sample_base > 0 else math.inf

            ci = None
            if args.bootstrap_trials > 0:
                if zone_name == "within_p_1pct":
                    flags = [_within_relative(e.candidate, p_int, 1) for e in slice_entries]
                elif zone_name == "within_q_1pct":
                    flags = [_within_relative(e.candidate, q_int, 1) for e in slice_entries]
                else:
                    flags = [
                        _within_relative(e.candidate, p_int, 5) or _within_relative(e.candidate, q_int, 5)
                        for e in slice_entries
                    ]
                ci = _bootstrap_enrichment(
                    flags,
                    expected_base,
                    rng=bootstrap_rng,
                    trials=args.bootstrap_trials,
                )

            k_metrics[zone_name] = {
                "topk": k,
                "top_frac": top_frac,
                "baseline_expected": expected_base,
                "baseline_sample": sample_base,
                "enrichment_expected": enrich_expected,
                "enrichment_sample": enrich_sample,
                "bootstrap_ci_enrichment_expected": list(ci) if ci is not None else None,
            }

        metrics[str(k)] = k_metrics

    # Statistical tests: compare Top-1000 vs random candidates (same window).
    stats_out: dict[str, object] = {"skipped": True}
    if not args.no_stats and scipy_stats is not None and 1_000 <= len(top_sorted):
        baseline_rng = random.Random(args.seed + 2)
        random_candidates = [win_min_odd + (2 * baseline_rng.randrange(total_odds)) for _ in range(1_000)]

        def signed_pos(candidate: int) -> float:
            return float(candidate - sqrt_n)

        def nearest_factor_dist(candidate: int) -> float:
            return float(min(abs(candidate - p_int), abs(candidate - q_int)))

        top_positions = [signed_pos(e.candidate) for e in top_sorted[:1_000]]
        rnd_positions = [signed_pos(c) for c in random_candidates]

        top_nearest = [nearest_factor_dist(e.candidate) for e in top_sorted[:1_000]]
        rnd_nearest = [nearest_factor_dist(c) for c in random_candidates]

        ks = scipy_stats.ks_2samp(top_positions, rnd_positions)
        mw = scipy_stats.mannwhitneyu(top_nearest, rnd_nearest, alternative="less")

        stats_out = {
            "skipped": False,
            "ks_2samp_signed_distance_from_sqrt": {
                "statistic": float(ks.statistic),
                "pvalue": float(ks.pvalue),
            },
            "mannwhitneyu_nearest_factor_distance": {
                "statistic": float(mw.statistic),
                "pvalue": float(mw.pvalue),
                "alternative": "less",
            },
        }

    summary = {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host_platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "gmpy2": getattr(gmpy2, "__version__", None),
            "scipy": getattr(scipy, "__version__", None) if scipy is not None else None,
            "num_candidates": num_candidates,
            "seed": args.seed,
            "chunk_size": args.chunk_size,
            "processes": args.processes,
            "topk_max": topk_max,
            "bootstrap_trials": args.bootstrap_trials,
            "elapsed_seconds": elapsed,
            "csv_path": str(out_csv_path) if out_csv_path is not None else None,
        },
        "constants": {
            "N_127": str(N_127),
            "p": str(P_127),
            "q": str(Q_127),
            "sqrt_n": str(sqrt_n),
            "window_radius_pct": 13,
            "search_min": str(search_min),
            "search_max": str(search_max),
            "window_min_odd": str(win_min_odd),
            "window_max_odd": str(win_max_odd),
            "total_odds_in_window": total_odds,
        },
        "baseline_expected": baseline_expected,
        "baseline_sample": baseline_sample,
        "metrics": metrics,
        "stats": stats_out,
    }

    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        f"Done: scored {num_candidates:,} candidates in {elapsed:.2f}s; "
        f"Top-K retained: {len(top_sorted):,}.",
        file=sys.stderr,
    )
    print(f"Wrote summary JSON: {args.summary_json}", file=sys.stderr)
    if out_csv_path is not None:
        print(f"Wrote CSV: {out_csv_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
