# Guidelines for AI Code Assistants

**READ THIS FIRST** before generating or reviewing any code in this repository.

---

## ⛔ CRITICAL RULE: 64-BIT INTEGER TYPES ARE STRICTLY FORBIDDEN

### Banned Data Types

The following data types are **NEVER** permitted anywhere in this codebase:

- `np.int64` / `numpy.int64`
- `np.uint64` / `numpy.uint64`  
- Any NumPy dtype that defaults to 64-bit integers
- C/C++ `int64_t`, `uint64_t`, `long long`
- Any fixed-width 64-bit integer representation

**NO EXCEPTIONS.** Not in intermediate calculations, not in array indexing, not in "temporary" variables, not anywhere.

---

## Why This Rule Exists

### The Problem: Silent Integer Overflow

This repository works with **127-bit semiprimes** and factorization candidates near √N ≈ 1.17 × 10³⁸.

**`np.int64` has a maximum value of 9,223,372,036,854,775,807 (≈ 9.2 × 10¹⁸).**

When you cast or add values exceeding this limit to `np.int64`:

1. **NumPy silently wraps the value** into the int64 range via modular arithmetic
2. **No error is raised, no warning is issued**
3. The resulting value is **completely wrong** but looks like valid data
4. Your algorithm continues executing with garbage values
5. **The search explores the wrong region of the factorization space entirely**

### Real Example from PR #7 (BROKEN CODE)

```python
# ❌ BROKEN - DO NOT COPY THIS
SEARCH_MIN = 105543860650446645996  # ≈ 1.06 × 10²⁰ (exceeds int64 max!)
width = SEARCH_MAX - SEARCH_MIN      # ≈ 2.35 × 10¹⁸  
d = SEARCH_MIN + (u * width).astype(np.int64)  # 💥 SILENT OVERFLOW
```

**What happens:**
- `SEARCH_MIN` (1.06 × 10²⁰) > `np.iinfo(np.int64).max` (9.2 × 10¹⁸)
- When NumPy broadcasts the addition, it wraps `SEARCH_MIN` into the int64 range
- The resulting candidates `d` are **negative numbers or small positive integers**
- The GCD test runs against **completely wrong values**, nowhere near √N
- The 30-minute search tests **random junk instead of the target window**
- **The experiment silently fails with zero chance of finding the factor**

This is a **catastrophic silent failure**—the code runs to completion, reports "no factor found", but never actually searched the intended space.

---

## ✅ Required Data Types

Use **arbitrary-precision integers** exclusively:

### Python Native
```python
# ✅ CORRECT - Python int has unlimited precision
candidate = int(SEARCH_MIN) + offset  
```

### gmpy2 (Preferred for Performance)
```python
import gmpy2

# ✅ CORRECT - gmpy2.mpz handles arbitrary precision
candidate = gmpy2.mpz(SEARCH_MIN) + offset
g = gmpy2.gcd(candidate, N_127)
```

### NumPy Arrays (When Absolutely Necessary)
```python
# ✅ CORRECT - Use object dtype for arbitrary-precision integers
import numpy as np
import gmpy2

candidates = np.empty(batch_size, dtype=object)
for i in range(batch_size):
    candidates[i] = gmpy2.mpz(SEARCH_MIN + offsets[i])
```

**Note:** Using `dtype=object` disables NumPy's vectorized operations, but **correctness > performance**. You're working with 127-bit numbers—the arithmetic **must** be arbitrary-precision.

---

## Code Review Checklist

Before approving any code that handles factorization candidates or large integers:

- [ ] **No `np.int64`, `np.uint64`, or similar fixed-width types**
- [ ] **All arithmetic uses `gmpy2.mpz` or Python `int`**  
- [ ] **No `.astype(np.int64)` casts anywhere**
- [ ] **Array dtypes are `object` when storing large integers**
- [ ] **Verify SEARCH_MIN/SEARCH_MAX exceed int64 max—they should!**
- [ ] **Test with actual 127-bit values, not toy examples**

---

## For AI Code Assistants

If you are an LLM (Copilot, ChatGPT, Claude, etc.) generating code suggestions:

1. **Assume all integers in this repo exceed 64-bit range unless proven otherwise**
2. **Default to `gmpy2.mpz` for all candidate values, factors, and large constants**
3. **Flag any `np.int64` usage as a critical error in code review**
4. **When suggesting NumPy optimizations, verify they don't introduce fixed-width types**
5. **If unsure, prefer Python `int` over NumPy dtypes**

### Suggested Review Comments

When you detect int64 usage:

> ⚠️ **CRITICAL: int64 overflow risk**  
> This code uses `np.int64` with values exceeding 9.2×10¹⁸, which will cause silent wraparound. The search will test wrong candidates and fail silently. Replace with `gmpy2.mpz` or Python `int`.

---

## Summary

**TL;DR for agents:**

- This repo factors 127-bit semiprimes (√N ≈ 10³⁸)
- `np.int64` max is 9.2 × 10¹⁸—**way too small**
- Silent overflow produces garbage values  
- **Use `gmpy2.mpz` or Python `int` exclusively**
- **Flag all int64/uint64 usage as critical bugs**

If you generate code with `np.int64` for large integers in this repo, **you will break the factorization experiments silently**. Don't do it.
