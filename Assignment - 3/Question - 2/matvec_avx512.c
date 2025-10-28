#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>  // For AVX intrinsics

// Simple timing routine
double get_time_millis() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000.0) + (t.tv_nsec / 1.0e6);
}

// Naive (non-vectorized) matrix–vector multiplication
void matvec_naive(const float *A, const float *x, float *y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

// AVX-512–accelerated matrix–vector multiplication
// This assumes N is a multiple of 16 for simplicity.
// If N is not multiple of 16, handle the leftover in a scalar loop.
void matvec_avx512(const float *A, const float *x, float *y, int M, int N) {
    for (int i = 0; i < M; i++) {
        __m512 sum_vec = _mm512_setzero_ps();  // holds partial sums in 16 lanes
        int j = 0;
        // Process 16 floats at a time
        for (; j + 16 <= N; j += 16) {
            __m512 a_vec = _mm512_loadu_ps(&A[i * N + j]);
            __m512 x_vec = _mm512_loadu_ps(&x[j]);
            // Fused multiply-add: sum_vec += a_vec * x_vec
            sum_vec = _mm512_fmadd_ps(a_vec, x_vec, sum_vec);
        }
        // Horizontal sum of sum_vec's 16 lanes
        float partial[16];
        _mm512_storeu_ps(partial, sum_vec);
        float total = 0.0f;
        for (int k = 0; k < 16; k++) {
            total += partial[k];
        }
        // Handle any leftover elements if N not multiple of 16
        for (; j < N; j++) {
            total += A[i * N + j] * x[j];
        }
        y[i] = total;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s M N\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    // Allocate memory
    float *A = (float *) aligned_alloc(64, M * N * sizeof(float));
    float *x = (float *) aligned_alloc(64, N * sizeof(float));
    float *y_naive = (float *) aligned_alloc(64, M * sizeof(float));
    float *y_avx = (float *) aligned_alloc(64, M * sizeof(float));

    // Initialize inputs
    srand(0);
    for (int i = 0; i < M * N; i++) {
        A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < N; i++) {
        x[i] = (float)(rand() % 100) / 100.0f;
    }

    // Measure time for naive version
    double start = get_time_millis();
    matvec_naive(A, x, y_naive, M, N);
    double end = get_time_millis();
    double time_naive = end - start;

    // Measure time for AVX-512 version
    start = get_time_millis();
    matvec_avx512(A, x, y_avx, M, N);
    end = get_time_millis();
    double time_avx = end - start;

    // Compare correctness
    float max_diff = 0.0f;
    for (int i = 0; i < M; i++) {
        float diff = (y_naive[i] - y_avx[i]);
        if (diff < 0.0f) diff = -diff;
        if (diff > max_diff) max_diff = diff;
    }

    // Print out results
    printf("Time naive (ms):    %f\n", time_naive);
    printf("Time AVX-512 (ms):  %f\n", time_avx);
    printf("Speedup:            %f\n", time_naive / time_avx);
    printf("Max diff:           %f\n", max_diff);

    // No explicit free calls here

    return 0;
}