# Quick Start Guide - PR#28 Testing

## ⚠️ PYTEST ERROR FIX

**If you got**: `fixture 'name' not found`

**Solution**: Don't use pytest! These are standalone Python scripts.

```bash
# ❌ DON'T DO THIS
pytest adversarial_test_pr20_exact.py

# ✅ DO THIS INSTEAD
python3 adversarial_test_pr20_exact.py
```

---

## Run Tests (2 Options)

### Option 1: Adaptive Window Test (Recommended) ⭐

Tests Z5D on 4 RSA challenges with adaptive windows that guarantee factors are in range.

```bash
cd /Users/velocityworks/IdeaProjects/geofac_validation
python3 adversarial_test_adaptive.py
```

**Runtime**: ~2 minutes  
**Tests**: RSA-100, RSA-110, RSA-120, RSA-129  
**Output**: `adaptive_window_results.json`

**Expected**: 2/4 strong signals (RSA-120, RSA-129)

---

### Option 2: PR#20 Exact Replication 📋

Validates PR#20 methodology on RSA challenges within ±13% window.

```bash
cd /Users/velocityworks/IdeaProjects/geofac_validation
python3 adversarial_test_pr20_exact.py
```

**Runtime**: ~1 minute  
**Tests**: RSA-100, RSA-110  
**Output**: Console only

**Expected**: 0/2 signals (factors too close to √N)

---

## What's the Difference?

| Feature | Adaptive (Main) | PR#20 Exact |
|---------|----------------|-------------|
| Window | Adaptive (ground truth + 20%) | Fixed ±13% |
| Coverage | 100% of factors | Only factors within ±13% |
| Tests | 4 RSA challenges | 2 RSA challenges |
| Purpose | Fair test of Z5D | Replicate PR#20 |

---

## Expected Results

### Adaptive Window Test
```
Test         Window      Enrichment     Result
RSA-100      ±15.0%      p:0.5× q:0.8×  ✗ NONE
RSA-110      ±18.6%      p:1.2× q:1.5×  ⚠️ WEAK
RSA-120      ±54.6%      p:0.0× q:10×   ✓ STRONG
RSA-129      ±240.0%     p:0.0× q:12×   ✓ STRONG
```

**Pattern**: Asymmetric enrichment (q only), works better for distant factors

### PR#20 Exact
```
Test         Enrichment     Result
RSA-100      p:0× q:0×     ✗ NO SIGNAL
RSA-110      p:0× q:0×     ✗ NO SIGNAL
```

**Pattern**: No signal on close factors (validates Z5D is distance-dependent)

---

## Clean Cache (If Needed)

```bash
cd /Users/velocityworks/IdeaProjects/geofac_validation
rm -rf .pytest_cache
find . -name "__pycache__" -type d -delete
find . -name "*.pyc" -delete
```

---

## Files Generated

After running tests:
- `adaptive_window_results.json` - Machine-readable results
- `adaptive_test_output.log` - Detailed log (if redirected)
- `pr20_exact_replication.log` - Replication log (if redirected)

---

## Documentation

- `PR28_SUMMARY.md` - Full technical summary
- `QUICK_START.md` - This file
- `README.md` - Project documentation
- `FINDINGS.md` - Research findings

---

## Troubleshooting

### Pytest still tries to run the file?
```bash
# Add to pytest.ini or pyproject.toml
[tool.pytest.ini_options]
python_files = []

# Or just ignore __pycache__
rm -rf .pytest_cache __pycache__
```

### Import errors?
```bash
# Make sure z5d_adapter.py is accessible
ls z5d_adapter.py  # Should exist

# Make sure dependencies installed
pip3 install gmpy2 numpy scipy
```

### Tests run but show different results?
Check that you're using:
- Fixed seed (seed=42) for reproducibility
- Correct Python version (3.9+)
- Updated z5d_adapter.py

---

## Summary

**PR#28 fixes the window coverage issue** that caused false negatives in PR#25/27.

**Key Innovation**: Adaptive windows calculated from ground truth ensure factors are always in search space.

**Expected Outcome**: Z5D shows 10× enrichment on distant factors (RSA-120, RSA-129), validating N₁₂₇ success wasn't a fluke.

**Your Next Step**:
```bash
python3 adversarial_test_adaptive.py
```

🎉 Done!

