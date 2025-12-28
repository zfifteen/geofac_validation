<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Z5D Candidate Filter System

## Technical Specification

**Document Type:** Filter Design Specification
**Date:** December 28, 2025
**Scope:** Mathematical Constraint Filters for Factor Candidate Validation

***

## 1. Overview

This specification defines the six filters applied to candidate primes during generation. Each filter exploits a mathematical property that genuine factors of N must satisfy. Candidates failing any filter are immediately rejected, avoiding wasted computation on impossible factors.

The filters are ordered by computational cost, with the cheapest filters applied first. This ordering ensures maximum rejection occurs before the expensive primality test, dramatically improving overall throughput.

***

## 2. Filter 1: Terminal Digit Compatibility

### 2.1 Mathematical Basis

When two numbers are multiplied, the last digit of the product depends only on the last digits of the multiplicands. Specifically, if p ends in digit A and q ends in digit B, then their product N ends in the last digit of A times B.

For odd primes greater than 5, the only possible terminal digits are 1, 3, 7, and 9. This means N (being the product of two such primes) can only end in certain digits, and furthermore, only certain pairs of factor terminal digits can produce each possible N terminal digit.

### 2.2 Valid Combinations

**If N ends in 1:** The factors must end in (1 and 1), (3 and 7), or (9 and 9).

**If N ends in 3:** The factors must end in (1 and 3) or (7 and 9).

**If N ends in 7:** The factors must end in (1 and 7) or (3 and 9).

**If N ends in 9:** The factors must end in (1 and 9), (3 and 3), or (7 and 7).

No other terminal digits are possible for products of two odd primes greater than 5.

### 2.3 Filter Behavior

During initialization, the generator examines N's terminal digit and records the set of valid terminal digits for factors. For example, if N ends in 3, valid candidate terminal digits are 1, 3, 7, and 9 (all appearing in at least one valid pair).

During generation, each candidate's terminal digit is checked against this set. A candidate ending in an invalid digit cannot possibly be a factor of N and is immediately rejected.

### 2.4 Computational Cost

This filter requires a single modulo-10 operation on the candidate, followed by a set membership check. The cost is effectively zero—a few nanoseconds on modern hardware.

### 2.5 Expected Rejection Rate

Since candidates are generated as odd numbers, their terminal digits are distributed among 1, 3, 5, 7, and 9. Candidates ending in 5 are always rejected (5 is the only prime ending in 5, and N is not divisible by 5). For other terminal digits, roughly 60% of odd candidates have incompatible terminal digits depending on N's specific terminal digit.

***

## 3. Filter 2: Digital Root Compatibility

### 3.1 Mathematical Basis

The digital root of a number is the single digit obtained by repeatedly summing the digits until only one digit remains. Mathematically, this equals the number modulo 9, with the special case that multiples of 9 have digital root 9 rather than 0.

Digital roots have a multiplicative property: the digital root of a product equals the digital root of the product of the individual digital roots. In other words, if p has digital root A and q has digital root B, then N has digital root equal to (A times B) reduced to a single digit via the modulo-9 rule.

### 3.2 Valid Combinations

For each possible digital root of N (values 1 through 9), only certain pairs of factor digital roots can produce it. For example:

**If N has digital root 1:** Valid factor digital root pairs are (1,1), (4,7), (7,4), and (8,8).

**If N has digital root 9:** Valid factor digital root pairs are (1,9), (9,1), (3,3), (6,6), and (9,9).

The complete mapping covers all nine possible N digital roots.

### 3.3 Filter Behavior

During initialization, the generator computes N's digital root and enumerates all valid factor digital root pairs. It then extracts the set of all individual digital roots appearing in any valid pair.

During generation, each candidate's digital root is computed and checked against this set. A candidate with an incompatible digital root cannot possibly be a factor of N and is rejected.

### 3.4 Computational Cost

This filter requires a single modulo-9 operation on the candidate, followed by a set membership check. Like the terminal digit filter, the cost is effectively zero.

### 3.5 Expected Rejection Rate

Digital roots are distributed approximately uniformly among values 1 through 9 for large numbers. Depending on N's digital root, between 3 and 7 digital root values are valid for candidates, yielding rejection rates between 22% and 67%. On average, roughly 44% of candidates are rejected by this filter alone.

***

## 4. Filter 3: Small Factor Exclusion

### 4.1 Mathematical Basis

If a semiprime N has no small prime factors, then neither of its prime factors can have small prime factors either—because the factors are themselves prime and thus have no factors other than 1 and themselves.

Conversely, if N is divisible by a small prime, then exactly one of p or q equals that small prime (since N is a semiprime with exactly two prime factors).

### 4.2 Filter Behavior

During initialization, the generator computes the greatest common divisor of N with the product of small primes (specifically, 2 × 3 × 5 × 7 × 11 × 13 × 17 × 19 × 23 = 223,092,870).

**If this GCD equals 1:** N has no small prime factors. Therefore, any candidate that shares a factor with the small primes product cannot be a factor of N. Such candidates are rejected.

**If this GCD exceeds 1:** N has at least one small prime factor. In this case, candidates may or may not share small factors, and this filter passes all candidates through without rejection.

### 4.3 Computational Cost

This filter requires computing the GCD of the candidate with a fixed constant. GCD computation runs in logarithmic time relative to the smaller operand. For the small primes product (a 9-digit number), this is extremely fast—typically under 100 nanoseconds.

### 4.4 Expected Rejection Rate

For random odd numbers in a typical search window, roughly 30% are divisible by at least one small prime (3, 5, 7, etc.). When N has no small factors, this 30% is entirely rejected. When N does have small factors, this filter provides no rejection. Since most cryptographic semiprimes have no small factors, the typical rejection rate is around 30%.

***

## 5. Filter 4: Primality Verification

### 5.1 Mathematical Basis

The factors of a semiprime are, by definition, prime numbers. Any composite candidate is automatically not a factor of N and can be rejected.

This is not a filter based on N's properties—it is a universal requirement that factors be prime. However, it is positioned fourth in the filter cascade because it is by far the most computationally expensive check.

### 5.2 Filter Behavior

Each candidate undergoes probabilistic primality testing using a two-phase approach:

**Phase One - Miller-Rabin Test:** The candidate is subjected to 64 rounds of the Miller-Rabin primality test using randomly chosen bases. Each round that the candidate passes reduces the probability of it being composite by a factor of 4. After 64 rounds, a composite number has less than 1 in 10^38 probability of falsely appearing prime.

**Phase Two - Strong Lucas Test:** The candidate is additionally subjected to the Strong Lucas probable prime test. This test catches certain composites that might slip through Miller-Rabin (though such cases are extremely rare). The combination of Miller-Rabin and Strong Lucas is known as the Baillie-PSW test and has no known counterexamples despite extensive searching.

### 5.3 Computational Cost

Each Miller-Rabin round requires modular exponentiation with exponent roughly half the size of the candidate. For a 100-digit candidate, 64 rounds require approximately 1 millisecond. For a 500-digit candidate, the cost rises to roughly 10-20 milliseconds. The Strong Lucas test adds comparable cost.

This filter dominates total generation time. All preceding filters exist primarily to reduce how many candidates reach this expensive check.

### 5.4 Expected Rejection Rate

By the prime number theorem, the density of primes near a number x is approximately 1/ln(x). For a 100-digit number, roughly 1 in 230 odd numbers is prime. For a 500-digit number, roughly 1 in 1150 odd numbers is prime.

However, because earlier filters have already rejected many composites (those with wrong terminal digits, wrong digital roots, or small factors), the rejection rate at this stage is somewhat lower—roughly 95-99% of candidates reaching this filter are composite and rejected.

***

## 6. Filter 5: Quadratic Residue Test

### 6.1 Mathematical Basis

For any prime p that does not divide N, the Legendre symbol (N|p) indicates whether N is a quadratic residue modulo p. The Legendre symbol takes one of three values:

**Value +1:** N is a quadratic residue mod p, meaning some integer x exists where x² ≡ N (mod p). This is consistent with p being a factor of N, but does not prove it.

**Value -1:** N is a quadratic non-residue mod p, meaning no integer x exists where x² ≡ N (mod p). This is mathematically inconsistent with p dividing N. If p divided N, then N ≡ 0 (mod p), and 0 is always a quadratic residue (0² = 0). Therefore, a Legendre symbol of -1 proves p does not divide N.

**Value 0:** This occurs only when p divides N, meaning p is actually a factor. This is the "jackpot" case.

### 6.2 Filter Behavior

For each candidate that has passed primality testing (confirmed as a probable prime), the generator computes the Legendre symbol of N with respect to the candidate.

**If the result is -1:** The candidate provably does not divide N. It is rejected.

**If the result is 0:** The candidate divides N. This is a confirmed factor (subject to the small probability that primality testing gave a false positive). The candidate is accepted and flagged as a likely true factor.

**If the result is +1:** The candidate might or might not divide N. It cannot be excluded by this filter and passes through for resonance scoring.

### 6.3 Computational Cost

Computing the Legendre symbol requires evaluating the Jacobi symbol (a generalization to odd composites, though we only apply it to primes), which involves modular reduction of N by the candidate followed by a computation similar in cost to GCD. For a 100-digit candidate, this takes roughly 1 microsecond—significantly cheaper than primality testing but more expensive than the simple arithmetic filters.

### 6.4 Expected Rejection Rate

For a random prime p not dividing N, the Legendre symbol is +1 or -1 with equal probability. Therefore, roughly 50% of prime candidates are rejected by this filter.

This is a substantial rejection rate, and importantly, it comes after primality testing. This means the filter eliminates half of the expensive-to-generate prime candidates before they consume resonance scoring resources.

### 6.5 Positioning Rationale

This filter must come after primality testing because the Legendre symbol is only meaningful when computed with respect to a prime. Computing it for composite candidates would yield misleading results (the Jacobi symbol for composites does not have the same divisibility implications).

***

## 7. Filter Cascade Summary

| Order | Filter Name | Mathematical Property | Cost | Typical Rejection |
| :-- | :-- | :-- | :-- | :-- |
| 1 | Terminal Digit | Last digit multiplication rule | ~1 nanosecond | ~60% |
| 2 | Digital Root | Modulo-9 multiplication rule | ~1 nanosecond | ~44% |
| 3 | Small Factor | GCD structure inheritance | ~100 nanoseconds | ~30% |
| 4 | Primality | Factor must be prime | ~1-20 milliseconds | ~95-99% |
| 5 | Quadratic Residue | Legendre symbol divisibility | ~1 microsecond | ~50% |


***

## 8. Cumulative Rejection Analysis

Consider 10,000 random odd numbers sampled from the search window:

**After Filter 1 (Terminal Digit):** Approximately 4,000 remain (60% rejected)

**After Filter 2 (Digital Root):** Approximately 2,200 remain (44% of remainder rejected)

**After Filter 3 (Small Factor):** Approximately 1,500 remain (30% of remainder rejected)

**After Filter 4 (Primality):** Approximately 15-75 remain (95-99% of remainder rejected, depending on magnitude)

**After Filter 5 (Quadratic Residue):** Approximately 7-37 remain (50% of remainder rejected)

The cascade reduces 10,000 initial samples to roughly 10-40 fully validated candidates. More importantly, only 1,500 candidates (15% of initial samples) undergo the expensive primality test, and only 15-75 candidates (0.15-0.75% of initial samples) undergo Legendre symbol computation.

***

## 9. False Negative Guarantee

A critical property of all filters is that they never reject true factors. Each filter is based on a mathematical property that genuine factors necessarily satisfy:

- A true factor must have a terminal digit compatible with N's factorization
- A true factor must have a digital root compatible with N's digital root
- A true factor of a semiprime with no small factors cannot itself have small factors
- A true factor is by definition prime
- A true factor divides N, so its Legendre symbol is 0 (not -1)

No true factor can fail any filter. The filters only reject candidates that are mathematically impossible as factors. This guarantee is essential—false negatives would cause the system to miss valid factorizations.

***

## 10. False Positive Acceptance

Filters do not guarantee that passing candidates are true factors. They only eliminate impossible candidates. A candidate can pass all filters yet still not divide N.

**Terminal Digit:** Many non-factors share terminal digits with true factors

**Digital Root:** Many non-factors share digital roots with true factors

**Small Factor:** Non-factors may also lack small factors

**Primality:** Many primes exist that do not divide N

**Quadratic Residue:** Approximately half of all primes have N as a quadratic residue, regardless of whether they divide N

Verification that a candidate actually divides N requires trial division (computing N mod candidate and checking for zero remainder). This verification is performed downstream by the resonance scoring system, not by the filter cascade.

***

## 11. Filter Independence

The five filters test independent mathematical properties. Passing one filter provides no information about likelihood of passing other filters (with minor statistical correlations).

This independence is valuable because it means rejection rates compound multiplicatively. If Filter A rejects 60% and Filter B rejects 44%, together they reject approximately 78% (1 - 0.4 × 0.56), not 60% or 44%.

The exception is Filter 5 (Quadratic Residue), which can only be applied after Filter 4 (Primality) confirms the candidate is prime. These two filters have a sequential dependency, though the properties they test are still mathematically independent.

***

## 12. Tuning Considerations

### 12.1 Primality Test Strength

The 64 Miller-Rabin rounds provide overwhelming confidence but could be reduced to 40 rounds with still-negligible false positive risk. Reducing rounds decreases primality testing cost by roughly 35% at the expense of marginally increased false positive probability (still astronomically small at 1 in 10^24).

### 12.2 Small Primes Limit

The default small primes product uses primes through 23. Extending to primes through 29 or 31 marginally increases rejection rate at negligible cost increase. Reducing to primes through 13 slightly decreases rejection rate and cost.

### 12.3 Filter Disabling

Individual filters can be disabled for experimentation. Disabling cheap filters has minimal performance impact but reduces rejection before primality testing. Disabling the Legendre filter passes more candidates to resonance scoring, useful for studying score distributions across a broader candidate pool.

***

## 13. Implementation Requirements

### 13.1 Precision Preservation

All filter computations must maintain arbitrary precision for N and candidates. The terminal digit and digital root filters operate on single-digit remainders and pose no precision risk. The small factor GCD filter uses a fixed constant. The primality and Legendre filters require modular arithmetic on full-precision values and must use arbitrary-precision integer libraries.

### 13.2 Determinism

Given identical inputs, filters must produce identical accept/reject decisions. The only source of non-determinism is the random base selection in Miller-Rabin testing. When reproducibility is required, the random seed must be fixed during initialization.

### 13.3 Statistics Collection

Each filter should increment rejection counters to support diagnostic analysis. These counters reveal which filters provide the most rejection and help identify potential tuning opportunities.

