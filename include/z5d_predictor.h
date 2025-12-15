/**
 * Z5D nth-Prime Predictor - Public API
 * =====================================
 * 
 * High-precision C/MPFR implementation of the calibrated Z5D nth-prime
 * predictor (PNT + calibrated d/e terms) plus a discrete refinement layer
 * that guarantees a probable-prime output.
 * 
 * @file z5d_predictor.h
 * @version 1.0
 * @author Unified Framework Team
 */

#ifndef Z5D_PREDICTOR_H
#define Z5D_PREDICTOR_H

#include <stdio.h>
#include <stdint.h>
#include <mpfr.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Library version */
#define Z5D_PREDICTOR_VERSION "1.0.0"

/* Default precision in bits (equivalent to ~96 decimal places, comfortable for 10^12+) */
#define Z5D_DEFAULT_PRECISION 320

/* Kept for API compatibility (unused in closed-form path) */
#define Z5D_DEFAULT_K 10

/**
 * Result structure for nth prime prediction
 */
typedef struct {
    mpfr_t predicted_prime;  /* Predicted value (rounded MPFR) */
    mpfr_t error;            /* Reserved / unused in closed-form path */
    double elapsed_ms;       /* Computation time in milliseconds */
    int iterations;          /* Iterations performed (1 for closed form) */
    int converged;           /* 1 if completed prediction */
} z5d_result_t;

/**
 * Configuration for predictor
 */
typedef struct {
    mpfr_prec_t precision;   /* MPFR precision in bits */
    int K;                   /* Number of terms in R(x) series */
    int max_iterations;      /* Maximum Newton-Halley iterations */
    mpfr_t tolerance;        /* Convergence tolerance */
} z5d_config_t;

/**
 * Initialize library (must be called before using library)
 */
void z5d_init(void);

/**
 * Cleanup library (call when done)
 */
void z5d_cleanup(void);

/**
 * Get library version string
 */
const char* z5d_get_version(void);

/**
 * Initialize configuration with default values
 * 
 * @param config Configuration structure to initialize
 */
void z5d_config_init(z5d_config_t* config);

/**
 * Cleanup configuration
 * 
 * @param config Configuration structure to cleanup
 */
void z5d_config_clear(z5d_config_t* config);

/**
 * Initialize result structure
 * 
 * @param result Result structure to initialize
 * @param precision MPFR precision to use
 */
void z5d_result_init(z5d_result_t* result, mpfr_prec_t precision);

/**
 * Cleanup result structure
 * 
 * @param result Result structure to cleanup
 */
void z5d_result_clear(z5d_result_t* result);

/**
 * Predict the nth prime (approximate MPFR value) using default configuration
 * 
 * @param result Output result structure (must be initialized)
 * @param n Index of prime to predict (n >= 1)
 * @return 0 on success, negative on error
 */
int z5d_predict_nth_prime(z5d_result_t* result, uint64_t n);

/**
 * Predict the nth prime with custom configuration (approximate MPFR value)
 * 
 * @param result Output result structure (must be initialized)
 * @param n Index of prime to predict (n >= 1)
 * @param config Custom configuration
 * @return 0 on success, negative on error
 */
int z5d_predict_nth_prime_ex(z5d_result_t* result, uint64_t n, const z5d_config_t* config);

/**
 * Predict the nth prime and return a GMP integer refined to a probable prime.
 * This is the preferred entry point for exact comparisons / compliance tests.
 *
 * @param prime_out Output mpz_t (must be initialized by caller)
 * @param n Index of prime to predict (n >= 1)
 * @return 0 on success, negative on error
 */
int z5d_predict_nth_prime_mpz(mpz_t prime_out, uint64_t n);

/* Big-n entry points */
int z5d_predict_nth_prime_mpz_big(mpz_t prime_out, const mpz_t n);
int z5d_predict_nth_prime_str(mpz_t prime_out, const char* n_dec_str);

/* Legacy helpers retained for compatibility with existing math utilities */
int z5d_mobius(int n);
void z5d_riemann_R(mpfr_t rop, const mpfr_t x, int K, mpfr_prec_t prec);
void z5d_riemann_R_prime(mpfr_t rop, const mpfr_t x, int K, mpfr_prec_t prec);

#ifdef __cplusplus
}
#endif

#endif /* Z5D_PREDICTOR_H */
