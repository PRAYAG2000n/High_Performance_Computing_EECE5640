#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while(0)

__global__ void leibniz_kernel_float(float *partial_sums, unsigned long long iterations)
{
    unsigned long long idx    = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (unsigned long long k = idx; k < iterations; k += stride) {
        // Term = 4 * (-1)^k / (2k + 1)
        float sign = ((k & 1ULL) == 0ULL) ? 1.0f : -1.0f;
        local_sum += sign * (4.0f / (2.0f * k + 1.0f));
    }

    partial_sums[idx] = local_sum;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_iterations>\n", argv[0]);
        return 1;
    }

    unsigned long long n = atoll(argv[1]);
    if (n < 1) {
        fprintf(stderr, "Number of iterations must be >= 1.\n");
        exit(EXIT_FAILURE);
    }

    printf("Using single-precision calculation for %llu iterations.\n", n);

    int blockSize = 256;
    int gridSize  = 256;
    
    float *d_partial_sums;
    size_t totalThreads = (size_t)blockSize * gridSize;
    CHECK_CUDA( cudaMalloc((void**)&d_partial_sums, totalThreads * sizeof(float)) );

    leibniz_kernel_float<<<gridSize, blockSize>>>(d_partial_sums, n);
    CHECK_CUDA( cudaDeviceSynchronize() );

    float *h_partial_sums = (float*)malloc(totalThreads * sizeof(float));
    CHECK_CUDA( cudaMemcpy(h_partial_sums, d_partial_sums, totalThreads * sizeof(float), cudaMemcpyDeviceToHost) );

    float pi_approx = 0.0f;
    for (size_t i = 0; i < totalThreads; ++i) {
        pi_approx += h_partial_sums[i];
    }

    printf("Approximated value of PI = %.9f\n", pi_approx);

    free(h_partial_sums);
    CHECK_CUDA( cudaFree(d_partial_sums) );

    return 0;
}
