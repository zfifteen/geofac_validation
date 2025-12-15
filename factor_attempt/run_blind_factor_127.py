#!/usr/bin/env python3
import sys
import os
import time
import math
import numpy as np
from datetime import datetime, timezone

# Add parent directory and tools directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
tools_dir = os.path.join(repo_root, "tools")

if repo_root not in sys.path:
    sys.path.append(repo_root)
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

# Import Z5D adapter
try:
    from z5d_adapter import z5d_n_est, compute_z5d_score
except ImportError:
    print("Error: Could not import z5d_adapter. Make sure it is in the parent directory.")
    sys.exit(1)

# Import QMC generator
try:
    # Try importing as module from tools package if implicit namespace works
    from tools.generate_qmc_seeds import generate_sobol_sequence
except ImportError:
    try:
        # Try importing directly if tools is in path
        from generate_qmc_seeds import generate_sobol_sequence
    except ImportError:
        print("Error: Could not import generate_qmc_seeds.")
        sys.exit(1)

import config_127

def generate_batch(batch_id, batch_size):
    """
    Generate a batch of candidate integers using QMC.
    """
    # Use existing QMC generator (Sobol)
    # generate_sobol_sequence returns shape (num_samples, dimensions)
    # We use dimensions=1 and seed=batch_id for variation
    samples = generate_sobol_sequence(num_samples=batch_size, dimensions=1, seed=batch_id)
    u = samples[:, 0] # Extract 1D array
    
    width = config_127.SEARCH_MAX - config_127.SEARCH_MIN
    
    # Map to integers carefully to preserve precision
    # offset = int(u * width)
    # d = SEARCH_MIN + offset
    candidates = []
    for val in u:
        offset = int(val * width)
        d = config_127.SEARCH_MIN + offset
        candidates.append(d)
        
    return candidates

def score_batch(candidates):
    """
    Score candidates using Z5D predictor.
    Returns list of (score, candidate).
    """
    scores = []
    for d in candidates:
        d_str = str(d)
        # Estimate prime index n
        n_est = z5d_n_est(d_str)
        # Compute Z5D score (log relative deviation)
        score = compute_z5d_score(d_str, n_est)
        scores.append((score, d))
    return scores

def main():
    print(f"Starting blind factor attempt for N_127 = {config_127.N_127}")
    print(f"Bits: {config_127.BITS}")
    print(f"Search range: [{config_127.SEARCH_MIN}, {config_127.SEARCH_MAX}]")
    print(f"Total candidates: {config_127.TOTAL_CANDIDATES}")
    print(f"Time limit: {config_127.MAX_WALLCLOCK_SECONDS} seconds")
    
    start_time = time.time()
    batch_size = config_127.TOTAL_CANDIDATES // config_127.NUM_BATCHES
    
    best_candidates = [] # List of (score, d)
    
    for batch_idx in range(config_127.NUM_BATCHES):
        elapsed = time.time() - start_time
        if elapsed > config_127.MAX_WALLCLOCK_SECONDS:
            print("\nTime limit exceeded.")
            break
            
        print(f"Batch {batch_idx + 1}/{config_127.NUM_BATCHES} ({batch_size} candidates)...", end="\r", flush=True)
        
        candidates = generate_batch(batch_idx, batch_size)
        scored = score_batch(candidates)
        
        # Keep top K from this batch
        # Lower (more negative) score is better
        scored.sort(key=lambda x: x[0]) 
        top_k = scored[:config_127.TOP_K_PER_BATCH]
        
        # Check for factors
        for score, d in top_k:
            # We also check if d divides N. 
            # Note: d could be 1 or N, ignore those (though unlikely in this range)
            g = math.gcd(config_127.N_127, d)
            if g > 1 and g < config_127.N_127:
                print(f"\n\n!!! FOUND FACTOR !!!")
                print(f"Factor p: {g}")
                print(f"Factor q: {config_127.N_127 // g}")
                print(f"Found in batch {batch_idx+1}")
                print(f"Z5D Score: {score}")
                print(f"Time elapsed: {time.time() - start_time:.2f}s")
                return
        
        # Track global best for reporting
        best_candidates.extend(top_k)
        best_candidates.sort(key=lambda x: x[0])
        best_candidates = best_candidates[:config_127.TOP_K_PER_BATCH]
        
    final_elapsed_time = time.time() - start_time
    print(f"\n\nSearch complete. No factor found. Total elapsed time: {final_elapsed_time:.2f}s")
    if best_candidates:
        print(f"Best score seen: {best_candidates[0][0]}")
        print(f"Candidate: {best_candidates[0][1]}")

if __name__ == "__main__":
    main()