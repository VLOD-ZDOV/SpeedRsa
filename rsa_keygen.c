// rsa_keygen_optimized.c
// Optimized RSA 2048-bit key generation with cache awareness and parallelism

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <time.h>
#include <stdint.h>
#include <x86intrin.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <pthread.h>

#define LOG(...) fprintf(stderr, __VA_ARGS__)
#define RSA_BITS 2048
#define RSA_E 65537
#define CACHE_SIZE (32 * 1024) // L1 cache size
#define MILLER_RABIN_ITERS 6   // Reduced for speed

// Timer using rdtsc
uint64_t rdtsc() {
    return __rdtsc();
}

// Fast divisibility check for small primes (C instead of assembly to avoid SIGFPE)
static const unsigned small_primes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
static const size_t num_small_primes = sizeof(small_primes) / sizeof(small_primes[0]);

int is_divisible_by_small(mpz_t n) {
    fprintf(stderr, "[DEBUG] Entering is_divisible_by_small\n");
    if (mpz_cmp_ui(n, 2) <= 0) {
        fprintf(stderr, "[DEBUG] n <= 2, returning 1\n");
        return 1;
    }
    if (mpz_even_p(n)) {
        fprintf(stderr, "[DEBUG] n is even, returning 1\n");
        return 1;
    }

    for (size_t i = 0; i < num_small_primes; i++) {
        unsigned long div = small_primes[i];
        if (mpz_divisible_ui_p(n, div)) {
            fprintf(stderr, "[DEBUG] n divisible by %lu, returning 1\n", div);
            return 1;
        }
    }
    fprintf(stderr, "[DEBUG] Exiting is_divisible_by_small, not divisible\n");
    return 0;
}

// Miller-Rabin test (optimized)
int is_prime(mpz_t n, int reps, gmp_randstate_t *state) {
    fprintf(stderr, "[DEBUG] Entering is_prime with n = %Zd\n", n);
    if (mpz_cmp_ui(n, 2) < 0) {
        fprintf(stderr, "[DEBUG] n < 2, returning 0\n");
        return 0;
    }
    if (mpz_even_p(n)) {
        fprintf(stderr, "[DEBUG] n is even, returning 0\n");
        return 0;
    }
    if (is_divisible_by_small(n)) {
        fprintf(stderr, "[DEBUG] n divisible by small prime, returning 0\n");
        return 0;
    }

    mpz_t d, a, x, n_minus_1, r;
    mpz_inits(d, a, x, n_minus_1, r, NULL);
    fprintf(stderr, "[DEBUG] Initialized GMP variables\n");

    mpz_sub_ui(n_minus_1, n, 1);
    fprintf(stderr, "[DEBUG] n_minus_1 = %Zd\n", n_minus_1);
    if (mpz_cmp_ui(n_minus_1, 0) <= 0) {
        fprintf(stderr, "[DEBUG] n_minus_1 <= 0, returning 0\n");
        mpz_clears(d, a, x, n_minus_1, r, NULL);
        return 0;
    }
    mpz_set(d, n_minus_1);
    fprintf(stderr, "[DEBUG] d = %Zd\n", d);

    int s = 0;
    while (mpz_even_p(d)) {
        fprintf(stderr, "[DEBUG] d = %Zd is even, dividing by 2\n", d);
        mpz_divexact_ui(d, d, 2);
        s++;
    }
    fprintf(stderr, "[DEBUG] After while: d = %Zd, s = %d\n", d, s);
    if (mpz_cmp_ui(d, 0) == 0) {
        fprintf(stderr, "[DEBUG] d == 0, returning 0\n");
        mpz_clears(d, a, x, n_minus_1, r, NULL);
        return 0;
    }

    for (int i = 0; i < reps; i++) {
        mpz_urandomm(a, *state, n_minus_1);
        mpz_add_ui(a, a, 1);
        fprintf(stderr, "[DEBUG] a = %Zd\n", a);
        if (mpz_cmp_ui(a, 1) <= 0 || mpz_cmp(a, n_minus_1) >= 0) {
            fprintf(stderr, "[DEBUG] Invalid a, continuing\n");
            continue;
        }
        mpz_powm(x, a, d, n);
        fprintf(stderr, "[DEBUG] x = %Zd after powm\n", x);
        if (mpz_cmp_ui(x, 1) == 0 || mpz_cmp(x, n_minus_1) == 0) {
            fprintf(stderr, "[DEBUG] x == 1 or x == n-1, continuing\n");
            continue;
        }
        int cont = 0;
        for (int j = 0; j < s; j++) {
            mpz_powm_ui(x, x, 2, n);
            fprintf(stderr, "[DEBUG] x = %Zd after powm_ui\n", x);
            if (mpz_cmp(x, n_minus_1) == 0) {
                cont = 1;
                break;
            }
        }
        if (!cont) {
            fprintf(stderr, "[DEBUG] Not prime, returning 0\n");
            mpz_clears(d, a, x, n_minus_1, r, NULL);
            return 0;
        }
    }

    fprintf(stderr, "[DEBUG] Prime found, returning 1\n");
    mpz_clears(d, a, x, n_minus_1, r, NULL);
    return 1;
}

// Struct for prime generation thread
typedef struct {
    mpz_t prime;
    int bits;
    gmp_randstate_t state;
} PrimeGenArgs;

// Generate a random prime (thread-safe)
void *generate_prime_thread(void *arg) {
    PrimeGenArgs *args = (PrimeGenArgs *)arg;
    fprintf(stderr, "[DEBUG] Entering generate_prime_thread\n");
    if (!args->prime || !args->state) {
        fprintf(stderr, "[ERROR] Invalid prime or state in thread\n");
        return NULL;
    }
    do {
        mpz_urandomb(args->prime, args->state, args->bits);
        mpz_setbit(args->prime, args->bits - 1); // Ensure high bit is set
        mpz_setbit(args->prime, 0);              // Ensure odd
        gmp_fprintf(stderr, "[DEBUG] Generated candidate = %Zd\n", args->prime);
        if (mpz_cmp_ui(args->prime, 0) <= 0) {
            fprintf(stderr, "[DEBUG] Skipping invalid candidate\n");
            continue;
        }
    } while (!is_prime(args->prime, MILLER_RABIN_ITERS, &args->state));
    fprintf(stderr, "[DEBUG] Exiting generate_prime_thread\n");
    return NULL;
}

void generate_prime(mpz_t prime, int bits) {
    gmp_randstate_t state;
    gmp_randinit_mt(state);
    unsigned long seed = (unsigned) time(NULL) ^ rdtsc();
    gmp_randseed_ui(state, seed);
    fprintf(stderr, "[DEBUG] Initialized state in generate_prime\n");

    PrimeGenArgs args;
    mpz_init(args.prime);
    args.bits = bits;
    gmp_randinit_mt(args.state);
    gmp_randseed_ui(args.state, seed ^ 1);
    fprintf(stderr, "[DEBUG] Initialized args in generate_prime\n");

    generate_prime_thread(&args);
    mpz_set(prime, args.prime);
    mpz_clear(args.prime);
    gmp_randclear(args.state);
    gmp_randclear(state);
}

int main() {
    LOG("[INFO] Generating 2048-bit RSA keypair (optimized)...\n");
    uint64_t start = rdtsc();

    // Cache-aligned buffer for temporary GMP data
    char *cache_buf = aligned_alloc(64, CACHE_SIZE);
    if (!cache_buf) {
        LOG("[ERROR] Failed to allocate cache buffer\n");
        return 1;
    }
    memset(cache_buf, 0, CACHE_SIZE);

    // Pre-allocate all GMP variables
    mpz_t p, q, n, phi, e, d, dP, dQ, qInv, p1, q1;
    mpz_inits(p, q, n, phi, e, d, dP, dQ, qInv, p1, q1, NULL);
    fprintf(stderr, "[DEBUG] Initialized GMP variables in main\n");

    // Parallel prime generation
    pthread_t threads[2];
    PrimeGenArgs args_p, args_q;

    mpz_init(args_p.prime);
    args_p.bits = RSA_BITS / 2;
    gmp_randinit_mt(args_p.state);
    gmp_randseed_ui(args_p.state, (unsigned) time(NULL) ^ rdtsc());
    fprintf(stderr, "[DEBUG] Initialized args_p\n");

    mpz_init(args_q.prime);
    args_q.bits = RSA_BITS / 2;
    gmp_randinit_mt(args_q.state);
    gmp_randseed_ui(args_q.state, (unsigned) time(NULL) ^ rdtsc() ^ 1);
    fprintf(stderr, "[DEBUG] Initialized args_q\n");

    pthread_create(&threads[0], NULL, generate_prime_thread, &args_p);
    pthread_create(&threads[1], NULL, generate_prime_thread, &args_q);
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
    fprintf(stderr, "[DEBUG] Threads joined\n");

    mpz_set(p, args_p.prime);
    mpz_set(q, args_q.prime);
    mpz_clear(args_p.prime);
    mpz_clear(args_q.prime);
    gmp_randclear(args_p.state);
    gmp_randclear(args_q.state);

    // Compute RSA parameters
    mpz_mul(n, p, q);
    mpz_sub_ui(p1, p, 1);
    mpz_sub_ui(q1, q, 1);
    mpz_mul(phi, p1, q1);
    fprintf(stderr, "[DEBUG] Computed RSA parameters\n");

    mpz_set_ui(e, RSA_E);
    if (mpz_invert(d, e, phi) == 0) {
        LOG("[ERROR] e has no modular inverse\n");
        free(cache_buf);
        mpz_clears(p, q, n, phi, e, d, dP, dQ, qInv, p1, q1, NULL);
        return 1;
    }

    mpz_mod(dP, d, p1);
    mpz_mod(dQ, d, q1);
    mpz_invert(qInv, q, p);

    uint64_t end = rdtsc();
    LOG("[RESULT] RSA key generated in %llu cycles\n", (unsigned long long)(end - start));

    // Export to PEM
    BIGNUM *bn_p = BN_new();
    BIGNUM *bn_q = BN_new();
    BIGNUM *bn_n = BN_new();
    BIGNUM *bn_e = BN_new();
    BIGNUM *bn_d = BN_new();
    BIGNUM *bn_dmp1 = BN_new();
    BIGNUM *bn_dmq1 = BN_new();
    BIGNUM *bn_iqmp = BN_new();

    BN_dec2bn(&bn_p, mpz_get_str(NULL, 10, p));
    BN_dec2bn(&bn_q, mpz_get_str(NULL, 10, q));
    BN_dec2bn(&bn_n, mpz_get_str(NULL, 10, n));
    BN_dec2bn(&bn_e, mpz_get_str(NULL, 10, e));
    BN_dec2bn(&bn_d, mpz_get_str(NULL, 10, d));
    BN_dec2bn(&bn_dmp1, mpz_get_str(NULL, 10, dP));
    BN_dec2bn(&bn_dmq1, mpz_get_str(NULL, 10, dQ));
    BN_dec2bn(&bn_iqmp, mpz_get_str(NULL, 10, qInv));

    RSA *rsa = RSA_new();
    RSA_set0_key(rsa, bn_n, bn_e, bn_d);
    RSA_set0_factors(rsa, bn_p, bn_q);
    RSA_set0_crt_params(rsa, bn_dmp1, bn_dmq1, bn_iqmp);

    FILE *priv = fopen("private_key.pem", "w");
    FILE *pub = fopen("public_key.pem", "w");
    PEM_write_RSAPrivateKey(priv, rsa, NULL, NULL, 0, NULL, NULL);
    PEM_write_RSA_PUBKEY(pub, rsa);
    fclose(priv);
    fclose(pub);
    LOG("[INFO] RSA keys exported to private_key.pem and public_key.pem\n");

    // Cleanup
    mpz_clears(p, q, n, phi, e, d, dP, dQ, qInv, p1, q1, NULL);
    RSA_free(rsa);
    free(cache_buf);
    return 0;
}
