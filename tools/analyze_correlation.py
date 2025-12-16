import sys
import json
import scipy.stats
import numpy as np

def main():
    data = []
    for line in sys.stdin:
        if not line.strip() or "_metadata" in line:
            continue
        try:
            d = json.loads(line)
            if "amplitude" in d and "z5d_score_p" in d:
                data.append(d)
        except:
            continue
            
    if not data:
        print("No valid data found.")
        sys.exit(1)
        
    amplitudes = [d["amplitude"] for d in data]
    scores = [d["z5d_score_p"] for d in data]
    
    r, p_val = scipy.stats.pearsonr(amplitudes, scores)
    
    print(f"Count: {len(data)}")
    print(f"Correlation (Amplitude vs Z5D Score): {r:.4f}")
    print(f"P-value: {p_val:.4e}")
    
    # Also check if top 10% amplitude have better scores than bottom 10%
    sorted_by_amp = sorted(data, key=lambda x: x["amplitude"])
    n = len(data)
    k = max(10, int(n * 0.1))
    
    bottom_k = sorted_by_amp[:k]
    top_k = sorted_by_amp[-k:]
    
    mean_score_bottom = np.mean([x["z5d_score_p"] for x in bottom_k])
    mean_score_top = np.mean([x["z5d_score_p"] for x in top_k])
    
    print(f"Mean Score (Bottom {k} amplitude): {mean_score_bottom:.4f}")
    print(f"Mean Score (Top {k} amplitude): {mean_score_top:.4f}")
    
    diff = mean_score_top - mean_score_bottom
    print(f"Difference (Top - Bottom): {diff:.4f} (Negative is better)")

if __name__ == "__main__":
    main()
