#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <mpfr.h>
#include "../include/z5d_predictor.h"

// Simple JSON parsing helper to extract string values
void extract_json_value(const char* json, const char* key, char* buffer, size_t max_len) {
    char search_key[256];
    sprintf(search_key, "\"%s\":", key);
    char* pos = strstr(json, search_key);
    if (!pos) {
        // Try with space
        sprintf(search_key, "\"%s\": ", key); 
        pos = strstr(json, search_key);
        if (!pos) {
            buffer[0] = '\0';
            return;
        }
    }
    pos += strlen(search_key);
    while (*pos == ' ' || *pos == '"') pos++; // Skip space/quote
    int i = 0;
    while (*pos != '"' && *pos != ',' && *pos != '}' && i < max_len - 1) {
        buffer[i++] = *pos++;
    }
    buffer[i] = '\0';
}

int main(int argc, char** argv) {
    z5d_init();
    
    // Config
    int dps = 512;
    // Simple arg parsing for dps
    for (int i=1; i<argc; i++) {
        if (strcmp(argv[i], "--dps") == 0 && i+1 < argc) {
            dps = atoi(argv[i+1]);
        }
    }
    
    mpfr_set_default_prec(dps);
    
    char line[65536]; // Large buffer for JSONL
    while (fgets(line, sizeof(line), stdin)) {
        if (strstr(line, "_metadata")) {
            printf("%s", line);
            continue;
        }
        
        char p_str[1024], q_str[1024];
        extract_json_value(line, "p", p_str, sizeof(p_str));
        extract_json_value(line, "q", q_str, sizeof(q_str));
        
        if (p_str[0] == '\0' || q_str[0] == '\0') {
            // Pass through if p or q not found
            printf("%s", line); 
            continue;
        }
        
        mpz_t p, q;
        mpz_init_set_str(p, p_str, 10);
        mpz_init_set_str(q, q_str, 10);
        
        // Ranking logic
        // 1. Estimate index n for p using R(p)
        // 2. Predict p' = Z5D(n)
        // 3. Score = |p - p'|
        
        mpfr_t p_mpfr, q_mpfr, k_est_p, k_est_q, diff_p, diff_q;
        mpfr_inits2(dps, p_mpfr, q_mpfr, k_est_p, k_est_q, diff_p, diff_q, (mpfr_ptr)0);
        
        mpfr_set_z(p_mpfr, p, MPFR_RNDN);
        mpfr_set_z(q_mpfr, q, MPFR_RNDN);
        
        // Inverse approx: R(x) ~ pi(x) ~ n
        z5d_riemann_R(k_est_p, p_mpfr, 10, dps); 
        z5d_riemann_R(k_est_q, q_mpfr, 10, dps);
        
        // For now, keep uint64 but cap properly
        // Use mpz for large n_est
        mpz_t n_p_mpz, n_q_mpz;
        mpz_init(n_p_mpz);
        mpz_init(n_q_mpz);
        mpfr_get_z(n_p_mpz, k_est_p, MPFR_RNDN);
        mpfr_get_z(n_q_mpz, k_est_q, MPFR_RNDN);
        
        z5d_result_t res_p, res_q;
        z5d_result_init(&res_p, dps);
        z5d_result_init(&res_q, dps);
        
        // For large n, z5d_predict_nth_prime may not handle, but assume it does or use approximation
        uint64_t n_p_small = mpz_get_ui(n_p_mpz);
        uint64_t n_q_small = mpz_get_ui(n_q_mpz);
        z5d_predict_nth_prime(&res_p, n_p_small);
        z5d_predict_nth_prime(&res_q, n_q_small);
        
        mpfr_sub(diff_p, p_mpfr, res_p.predicted_prime, MPFR_RNDN);
        mpfr_sub(diff_q, q_mpfr, res_q.predicted_prime, MPFR_RNDN);
        
        double score_p = mpfr_get_d(diff_p, MPFR_RNDN);
        if(score_p < 0) score_p = -score_p;
        
        double score_q = mpfr_get_d(diff_q, MPFR_RNDN);
        if(score_q < 0) score_q = -score_q;
        
        // Append to JSON line
        // Remove trailing newline and potential '}'
        size_t len = strlen(line);
        while(len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (line[len-1] == '}') line[--len] = '\0';
        
        // Output n_est as string
        char *n_p_str = mpz_get_str(NULL, 10, n_p_mpz);
        char *n_q_str = mpz_get_str(NULL, 10, n_q_mpz);
        printf("%s, \"z5d_score_p\": %.4f, \"z5d_n_est_p\": \"%s\", \"z5d_score_q\": %.4f, \"z5d_n_est_q\": \"%s\"}\n",
                line, score_p, n_p_str, score_q, n_q_str);
        free(n_p_str);
        free(n_q_str);
        
        mpfr_clears(p_mpfr, q_mpfr, k_est_p, k_est_q, diff_p, diff_q, (mpfr_ptr)0);
        mpz_clears(p, q, n_p_mpz, n_q_mpz, NULL);
        z5d_result_clear(&res_p);
        z5d_result_clear(&res_q);
    }
    
    z5d_cleanup();
    return 0;
}