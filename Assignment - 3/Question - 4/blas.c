#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>  // OpenBLAS CBLAS header

// Size of the square matrices
#define N 256

// Get current time in seconds for timing
double get_time_in_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

// Naive dense matrix multiplication (C = A * B)
void mm_naive(const float *A, const float *B, float *C, int n) {
    // C[i,j] = sum_k [ A[i,k] * B[k,j] ]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    // Allocate memory for the matrices
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C_naive = (float *)malloc(N * N * sizeof(float));
    float *C_blas  = (float *)malloc(N * N * sizeof(float));

    // Initialize A and B with random or dummy values
    // (Here we do something reproducible but you can use any approach.)
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)((i + 1) % 101) / 101.0f;  // Pseudo-random 0..1
        B[i] = (float)((i + 7) % 101) / 101.0f;  
    }

    // Time the naive matrix multiplication
    double start = get_time_in_seconds();
    mm_naive(A, B, C_naive, N);
    double end = get_time_in_seconds();
    double naive_time = end - start;
    printf("Naive multiplication time (seconds): %f\n", naive_time);

    // Time the OpenBLAS multiplication
    //   cblas_sgemm uses row-major by default if you use CblasRowMajor,
    //   so the parameters are (Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc).
    //   For 256×256 multiplication: M=N=K=256, alpha=1.0, beta=0.0.
    start = get_time_in_seconds();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0f,
                A, N,
                B, N,
                0.0f,
                C_blas, N);
    end = get_time_in_seconds();
    double blas_time = end - start;
    printf("OpenBLAS multiplication time (seconds): %f\n", blas_time);

    // Compare results (optional). Check if differences remain small.
    // This step is not strictly required, but it’s a good sanity check.
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = C_naive[i] - C_blas[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
    }
    printf("Maximum difference between naive and BLAS results: %g\n", max_diff);

    return 0;
}
