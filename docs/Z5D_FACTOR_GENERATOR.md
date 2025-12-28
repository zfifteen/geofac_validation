# Z5D Filtered Prime Generator for Balanced Semiprime Factorization

## Technical Specification v2.0

**Document Type:** High-Level Design Specification
**Date:** December 28, 2025
**Scope:** N-Aware Prime Candidate Generator with Integrated Constraint Filtering

***

## 1. Purpose

This specification defines a specialized variant of the Z5D prime generator designed specifically for factoring balanced semiprimes. The generator accepts a semiprime N during initialization and exposes a single method that produces paired factor candidates (p, q) on each invocation.

***

## 2. Core Concept

The generator is a stateful object that "learns" everything it can about N's factors before generating any candidates. When asked to produce candidates, it only emits prime pairs that are mathematically compatible with N's structure. Each call to the generation method returns a fresh pair of probable primes that could plausibly multiply to produce N.

***

## 3. Initialization Behavior

When the generator receives semiprime N, it performs a one-time analysis phase before any candidates are produced.

### 3.1 Precision Setup

The generator examines N's digit count and configures its internal arithmetic precision accordingly. For a 100-digit N, it allocates roughly 170 decimal places of precision. For a 617-digit N (magnitude 10^1233), it allocates roughly 946 decimal places. This ensures all subsequent calculations maintain full accuracy without floating-point truncation.

### 3.2 Window Computation

The generator computes the square root of N with full precision. It then defines a search window based on the target balance ratio of approximately 1.0558 (matching RSA-100's factor balance). The smaller factor p must fall between the square root divided by about 1.028 and the square root itself. The larger factor q must fall between the square root and the square root multiplied by about 1.028. This creates a narrow band where balanced factors must reside.

### 3.3 Terminal Digit Constraints

The generator examines the last digit of N. Based on this single digit, it determines which ending digits are valid for factors. For example, if N ends in 1, then valid factor pairs must end in (1,1), (3,7), (7,3), or (9,9). The generator stores the set of permissible ending digits for rapid filtering during candidate generation.

### 3.4 Digital Root Constraints

The generator computes N's digital root (the iterative sum of digits reduced to a single digit, equivalent to N modulo 9 with 0 mapped to 9). It then enumerates all pairs of digital roots whose product yields N's digital root. The generator stores the set of permissible digital roots for candidate factors.

### 3.5 Small Factor Analysis

The generator computes the greatest common divisor of N with the product of small primes (2 through 23). If this GCD equals 1, then N has no small prime factors, which means neither p nor q can have small prime factors either. The generator records this constraint for later filtering.

### 3.6 State Initialization

The generator initializes its internal random state for reproducible sampling if a seed is provided. It also initializes counters for tracking generation statistics including total attempts, rejection counts per filter, and successful generations.

***

## 4. Candidate Generation Method

The generator exposes a single method that produces one (p, q) candidate pair per invocation. Each call is independent—the generator does not remember or exclude previously returned candidates.

### 4.1 Method Signature

The method takes no arguments and returns a pair of probable primes. If the generator fails to produce a valid pair after a reasonable number of internal attempts, it returns an empty result or raises an exception indicating generation failure.

### 4.2 Generation Process for Each Candidate

The generator performs the following sequence for each of p and q independently:

**Position Sampling:** The generator samples a position within the appropriate window (p's window or q's window) using Z5D resonance-weighted sampling. Positions near geometric resonance peaks receive higher selection probability than arbitrary positions.

**Odd Enforcement:** If the sampled position is even, the generator increments it by one to make it odd. All prime factors of RSA semiprimes are odd.

**Terminal Digit Check:** The generator examines the candidate's last digit. If it does not appear in the precomputed set of valid terminal digits, the candidate is immediately rejected and a new position is sampled.

**Digital Root Check:** The generator computes the candidate's digital root. If it does not appear in the precomputed set of valid digital roots, the candidate is immediately rejected.

**Small Factor Check:** If N was determined to have no small prime factors, the generator checks whether the candidate shares any factors with the small primes product. If it does, the candidate is rejected.

**Primality Testing:** The generator subjects the candidate to probabilistic primality testing using Miller-Rabin with 64 rounds followed by a Strong Lucas test. This achieves negligible false positive probability. If the candidate fails primality testing, it is rejected.

**Quadratic Residue Check:** The generator computes the Legendre symbol of N with respect to the candidate. If the result is negative one, N is a quadratic non-residue modulo the candidate, which mathematically proves the candidate cannot divide N. Such candidates are rejected.

**Acceptance:** If the candidate passes all checks, it is accepted as a valid probable prime factor candidate.

### 4.3 Pair Construction

The method generates p and q independently using the process above. For p, it samples from the lower portion of the window (below the square root). For q, it samples from the upper portion (above the square root). Once both are successfully generated, the method returns the pair.

### 4.4 Complementary Relationship

Although p and q are generated independently, the generator ensures their product would fall within a reasonable range of N. Specifically, if p times q would differ from N by more than the expected tolerance for balanced factors, the pair may be regenerated. This optional consistency check prevents returning wildly incompatible pairs.

***

## 5. Filter Ordering Rationale

The filters are applied in strict order from cheapest to most expensive computational cost. This ordering maximizes throughput by rejecting invalid candidates as early as possible.

**First:** Terminal digit check requires a single modulo-10 operation. Cost is essentially zero.

**Second:** Digital root check requires a single modulo-9 operation. Cost is essentially zero.

**Third:** Small factor GCD check requires computing GCD with a precomputed constant. Cost is logarithmic in candidate size.

**Fourth:** Primality testing requires dozens of modular exponentiations. This is by far the most expensive check, consuming roughly 99% of total generation time.

**Fifth:** Legendre symbol check requires modular exponentiation but on a number already verified as prime. Cost is moderate but only applied to candidates that passed the expensive primality test.

By rejecting roughly 70-80% of candidates before primality testing, the filter cascade dramatically improves overall throughput.

***

## 6. Output Characteristics

### 6.1 Candidate Properties

Each returned p satisfies:
- Falls within the p-window (below square root of N)
- Ends in a digit compatible with N's terminal digit
- Has a digital root compatible with N's digital root
- Shares no small factors with N (when applicable)
- Is a probable prime with overwhelming probability
- Has N as a quadratic residue (Legendre symbol is not negative one)

Each returned q satisfies the same properties within the q-window (above square root of N).

### 6.2 No Duplicate Prevention

The generator does not track previously returned candidates. The same pair could theoretically be returned on multiple invocations, though this is statistically unlikely given the size of the search space. Duplicate prevention is the caller's responsibility if needed.

### 6.3 No Verification

The generator does not verify whether p times q equals N. It only ensures both candidates are plausible factors based on mathematical constraints. Verification via trial division is the caller's responsibility.

***

## 7. Statistics and Diagnostics

The generator maintains internal counters tracking:

- Total sampling attempts across all invocations
- Rejections at each filter stage (terminal digit, digital root, GCD, primality, Legendre)
- Successfully generated candidates
- Overall acceptance rate (successes divided by attempts)
- Per-filter rejection rates (rejections at each stage divided by total rejections)

These statistics are accessible via a separate method and provide insight into filter effectiveness and potential tuning opportunities.

***

## 8. Configuration Options

### 8.1 Balance Ratio

The expected ratio of q to p, defaulting to 1.0558. Adjusting this value widens or narrows the search windows accordingly.

### 8.2 Window Margin

An additional percentage buffer added to window bounds, defaulting to 1%. Provides safety margin against edge cases where true factors fall slightly outside theoretical bounds.

### 8.3 Primality Testing Strength

Number of Miller-Rabin rounds, defaulting to 64. Higher values reduce false positive probability at the cost of increased computation time.

### 8.4 Filter Toggles

Individual filters can be disabled for experimental purposes. For example, disabling the Legendre filter allows studying its impact on downstream resonance scoring accuracy.

### 8.5 Random Seed

Optional seed for reproducible candidate generation. When set, the same N with the same seed produces the same sequence of candidates.

***

## 9. Error Conditions

### 9.1 Invalid N

The generator rejects N values that are:
- Non-positive
- Even (cannot be product of two odd primes)
- Perfect squares (would imply p equals q)
- Too small (below minimum supported magnitude)
- Too large (above maximum supported magnitude for configured precision)

### 9.2 Generation Failure

If the generator fails to produce a valid candidate after a configured maximum number of attempts (default 1000 per candidate), it signals failure. This typically indicates misconfigured parameters or an N value with unusual properties.

### 9.3 Precision Inadequacy

If internal computations detect potential precision loss, the generator raises a warning or error. This can occur if N's magnitude exceeds the precision allocated during initialization.

***

## 10. Integration with Resonance Scoring

The generator is designed to feed candidates into the Geofac resonance scoring engine. The typical workflow is:

1. Initialize generator with N
2. Invoke generation method to obtain (p, q) pair
3. Score both p and q using resonance scorer
4. If either scores above threshold, verify via trial division
5. If verification fails, repeat from step 2
6. Continue until factor found or attempt budget exhausted

The generator and scorer operate independently—the generator knows nothing about resonance scores, and the scorer knows nothing about how candidates were generated. This separation of concerns allows independent optimization of each component.

***

## 11. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Initialization time | Under 100 milliseconds | One-time cost per N |
| Candidate pair generation | Under 50 milliseconds at 100-digit magnitude | Dominated by primality testing |
| Candidate pair generation | Under 500 milliseconds at 500-digit magnitude | Scales with digit count |
| Memory footprint | Under 100 megabytes | Stateless except for precomputed constraints |
| Filter rejection before primality | Greater than 70% | Measures filter cascade effectiveness |

***

## 12. Future Extensions

### 12.1 Adaptive Window Adjustment

If initial generations consistently fail verification, the generator could automatically widen its search windows to accommodate factors with unexpected balance ratios.

### 12.2 Learning from Failures

The generator could track which candidates came closest to dividing N (smallest remainder) and bias future sampling toward similar regions.

### 12.3 Parallel Generation

The generation method could accept a batch size parameter and return multiple pairs in parallel, leveraging multi-core processors for throughput improvement.

### 12.4 C Acceleration

Performance-critical filters (primality testing, Legendre symbol) could delegate to compiled C code using GMP/MPFR for significant speedup at extreme magnitudes.

***

## 13. Summary

This specification defines an N-aware prime generator that:

1. **Initializes once** with semiprime N, precomputing all derivable constraints
2. **Exposes a single method** that returns (p, q) candidate pairs on demand
3. **Applies cascading filters** ordered by computational cost to maximize throughput
4. **Guarantees mathematical compatibility** between returned candidates and N's structure
5. **Provides diagnostics** for monitoring filter effectiveness and generation statistics

The generator serves as the candidate source for downstream resonance scoring, producing only plausible factor candidates rather than arbitrary primes within a window.