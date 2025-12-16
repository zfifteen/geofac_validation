import json
import glob
from pathlib import Path

def main():
    files = glob.glob("data/*.jsonl")
    for file_path in files:
        with open(file_path, "r") as f:
            for line in f:
                if "_metadata" in line: continue
                try:
                    d = json.loads(line)
                    if "N" in d:
                        n_str = d["N"]
                        if len(n_str) > 1200 and len(n_str) < 1300:
                            print(f"Found candidate in {file_path}:")
                            print(f"Digits: {len(n_str)}")
                            print(json.dumps(d, indent=2))
                            return
                except:
                    continue
    print("No N ~ 10^1233 found.")

if __name__ == "__main__":
    main()
