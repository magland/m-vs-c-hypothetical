/*
 * lap2d_green.c -- C equivalent of lap2d_green.m
 *
 *   G(x,y) = -log(|x-y|^2) / (4*pi)
 *
 * Each rep k = 0..NREP-1 shifts target X coords by k*DT and adds the
 * resulting NT-by-NS kernel matrix into the accumulator `acc`.  The
 * final fingerprint is taken on `acc`, so every rep contributes to
 * the printed answer (no hoisting possible).
 *
 * Storage layout matches MATLAB column-major NT-by-NS:
 *   acc[j*NT + i]  for target i in [0,NT), source j in [0,NS).
 *
 * Build: see build.sh
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef NS
#define NS 4000
#endif
#ifndef NT
#define NT 4000
#endif
#ifndef NREP
#define NREP 5
#endif
#ifndef DT
#define DT 0.01
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __FAST_MATH__
#define BUILD_LABEL "fast-math"
#else
#define BUILD_LABEL "strict"
#endif

static inline uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static inline double xs64_double(uint64_t *s) {
    return (double)(xs64(s) >> 11) * (1.0 / 9007199254740992.0);
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Add -log(|t-s|^2)/(4*pi) (with target X shifted by k*dt) into acc. */
static void accumulate_kernel(const double *src_x, const double *src_y,
                              const double *tgt_x, const double *tgt_y,
                              double *acc, int k, double dt) {
    const double fourpi = 4.0 * M_PI;
    const double dx = (double)k * dt;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < NS; j++) {
        double sx = src_x[j], sy = src_y[j];
        double *acc_col = acc + (size_t)j * NT;
        for (int i = 0; i < NT; i++) {
            double rx = (tgt_x[i] + dx) - sx;
            double ry = tgt_y[i] - sy;
            double r2 = rx*rx + ry*ry;
            acc_col[i] += -log(r2) / fourpi;
        }
    }
}

int main(void) {
    static double src_x[NS], src_y[NS];
    static double tgt_x[NT], tgt_y[NT];

    size_t total = (size_t)NS * (size_t)NT;
    double *acc = (double *)malloc(total * sizeof(double));
    if (!acc) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* Inputs -- same xorshift64 seed and draw order as the .m file. */
    uint64_t state = 2463534242ULL;
    for (int j = 0; j < NS; j++) {
        src_x[j] = xs64_double(&state) * 2.0 - 1.0;
        src_y[j] = xs64_double(&state) * 2.0 - 1.0;
    }
    for (int i = 0; i < NT; i++) {
        tgt_x[i] = xs64_double(&state) * 2.0 - 1.0 + 3.0;
        tgt_y[i] = xs64_double(&state) * 2.0 - 1.0;
    }

    /* Warm up (one full pass through the kernel; result discarded). */
    memset(acc, 0, total * sizeof(double));
    accumulate_kernel(src_x, src_y, tgt_x, tgt_y, acc, 0, DT);

    /* Timed loop: accumulate NREP shifted kernel matrices into acc. */
    memset(acc, 0, total * sizeof(double));
    double t0 = now_seconds();
    for (int k = 0; k < NREP; k++) {
        accumulate_kernel(src_x, src_y, tgt_x, tgt_y, acc, k, DT);
    }
    double t1 = now_seconds();
    double elapsed = t1 - t0;

    /* Cross-language fingerprint. */
    int i_mid_m = NT / 2;
    int j_mid_m = NS / 2;
    int i_mid_c = i_mid_m - 1;
    int j_mid_c = j_mid_m - 1;

    double a_11  = acc[0];
    double a_mid = acc[(size_t)j_mid_c * NT + i_mid_c];
    double a_NN  = acc[(size_t)(NS-1) * NT + (NT-1)];

    double vmin = acc[0], vmax = acc[0], csum = 0.0;
    for (int j = 0; j < NS; j++) {
        const double *col = acc + (size_t)j * NT;
        for (int i = 0; i < NT; i++) {
            double v = col[i];
            csum += v;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
    }

    int nthreads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif

    printf("=== C lap2d_green ===\n");
    printf("build          = %s\n", BUILD_LABEL);
    printf("threads        = %d\n", nthreads);
    printf("NS=%d  NT=%d  NREP=%d  DT=%.4f\n", NS, NT, NREP, (double)DT);
    printf("elapsed_total  = %.6f s\n", elapsed);
    printf("elapsed_per    = %.6f s\n", elapsed / NREP);
    printf("throughput     = %.3e pts/s\n",
           (double)NS * (double)NT * (double)NREP / elapsed);
    printf("acc(1,1)       = %.17e\n", a_11);
    printf("acc(%d,%d) = %.17e\n", i_mid_m, j_mid_m, a_mid);
    printf("acc(NT,NS)     = %.17e\n", a_NN);
    printf("min(acc)       = %.17e\n", vmin);
    printf("max(acc)       = %.17e\n", vmax);
    printf("sum(acc)       = %.17e\n", csum);

    free(acc);
    return 0;
}
