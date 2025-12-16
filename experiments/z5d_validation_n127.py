#!/usr/bin/env python3
"""
Phase 1: Candidate Generation and Scoring for Z5D Validation (N127).
Generates 10,000 candidates around sqrt(N) and scores them.
"""

import sys
import json
import csv
import gmpy2
import mpmath
from pathlib import Path

# Add root to path for z5d_adapter
sys.path.append(str(Path(__file__).parent.parent))
try:
    import z5d_adapter
except ImportError:
    print("Error: Could not import z5d_adapter. Make sure you are in the project root or experiments folder.")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_FILE = Path(__file__).parent / "z5d_validation_n127_results.csv"
GROUND_TRUTH_FILE = DATA_DIR / "n127_ground_truth.json"

def generate_ground_truth():
    """
    Generate a 1233-digit semiprime N = p * q.
    p, q approx 10^616.
    """
    if GROUND_TRUTH_FILE.exists():
        print(f"Loading ground truth from {GROUND_TRUTH_FILE}")
        with open(GROUND_TRUTH_FILE, "r") as f:
            data = json.load(f)
            n_val = gmpy2.mpz(data["N"])
            p_val = gmpy2.mpz(data["p"])
            q_val = gmpy2.mpz(data["q"])
            return n_val, p_val, q_val

    print("Generating new N127 (1233-digit semiprime)...")
    rs = gmpy2.random_state(42)
    
    # Target 616 digits for p, q
    # 10^616
    
    base_p = gmpy2.mpz(10)**616
    offset_p = gmpy2.mpz_random(rs, 10**615)
    p = gmpy2.next_prime(base_p + offset_p)
    
    # Generate q
    base_q = gmpy2.mpz(10)**616
    offset_q = gmpy2.mpz_random(rs, 10**615)
    q = gmpy2.next_prime(base_q + offset_q)
    
    N = p * q
    
    data = {
        "N": str(N),
        "p": str(p),
        "q": str(q),
        "digits": len(str(N))
    }
    
    DATA_DIR.mkdir(exist_ok=True)
    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Generated N ({len(str(N))}) digits). Saved to {GROUND_TRUTH_FILE}")
    return N, p, q

def generate_candidates(N, p, q, num_candidates=10000):
    """
    Generate candidates in +/- 13% window around sqrt(N).
    """
    print(f"Generating {num_candidates} candidates...")
    
    sqrt_N = gmpy2.isqrt(N)
    
    # Window: +/- 13%
    range_width = int(sqrt_N * 0.13)
    
    low = sqrt_N - range_width
    high = sqrt_N + range_width
    span = high - low
    
    rs = gmpy2.random_state(123) 
    
    mpmath.mp.dps = 50
    sqrt_n_mpf = mpmath.mpf(str(sqrt_N))
    
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate", "z5d_score", "n_est", "dist_p", "dist_q", "pct_dev_sqrt"])
        
        for i in range(num_candidates):
            offset = gmpy2.mpz_random(rs, span)
            cand = low + offset
            
            # Ensure odd
            if cand % 2 == 0:
                cand += 1
                
            # Score
            cand_str = str(cand)
            try:
                n_est = z5d_adapter.z5d_n_est(cand_str)
                score = z5d_adapter.compute_z5d_score(cand_str, n_est)
                
                dist_p = abs(cand - p)
                dist_q = abs(cand - q)
                
                dev = (cand - sqrt_N)
                
                try:
                    dev_mpf = mpmath.mpf(str(dev))
                    pct_dev = float(dev_mpf / sqrt_n_mpf * 100.0)
                except:
                    pct_dev = 0.0
                
                writer.writerow([
                    cand_str,
                    f"{score:.6f}",
                    str(n_est),
                    str(dist_p),
                    str(dist_q),
                    f"{pct_dev:.6f}"
                ])
                
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            if i % 1000 == 0:
                print(f"Processed {i}/{num_candidates}...", end="\r")
                
    print(f"\nFinished. Results saved to {RESULTS_FILE}")

def main():
    N, p, q = generate_ground_truth()
    generate_candidates(N, p, q, num_candidates=10000)

if __name__ == "__main__":
    main()