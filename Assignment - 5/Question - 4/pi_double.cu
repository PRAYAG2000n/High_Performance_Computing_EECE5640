#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// CUDA error-checking macro (optional but helpful for debugging)
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while(0)

////////////////////////////////////////////////////////////////////////////////
// Kernel: each thread calculates a partial sum of the Leibniz series
////////////////////////////////////////////////////////////////////////////////
__global__ void leibniz_kernel_double(double *partial_sums, unsigned long long iterations)
{
    // Global thread index
    unsigned long long idx    = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    double local_sum = 0.0;

    // Each thread accumulates its portion of terms
    for (unsigned long long k = idx; k < iterations; k += stride) {
        // Term = 4 * (-1)^k / (2k + 1)
        double sign = ((k & 1ULL) == 0ULL) ? 1.0 : -1.0;
        local_sum += sign * (4.0 / (2.0 * k + 1.0));
    }

    // Store the thread's partial sum in global memory
    partial_sums[idx] = local_sum;
}

////////////////////////////////////////////////////////////////////////////////
// Host code
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_iterations>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Number of terms (iterations in the Leibniz series)
    unsigned long long n = atoll(argv[1]);
    if (n < 1) {
        fprintf(stderr, "Number of iterations must be >= 1.\n");
        exit(EXIT_FAILURE);
    }

    printf("Using double-precision calculation for %llu iterations.\n", n);

    // Choose a block size and grid size
    int blockSize = 256;
    int gridSize  = 256;  // Adjust as needed

    // Allocate space on the device for partial sums
    double *d_partial_sums;
    size_t totalThreads = (size_t)blockSize * gridSize;
    CHECK_CUDA( cudaMalloc((void**)&d_partial_sums, totalThreads * sizeof(double)) );

    // Launch the kernel
    leibniz_kernel_double<<<gridSize, blockSize>>>(d_partial_sums, n);
    CHECK_CUDA( cudaDeviceSynchronize() );

    // Copy partial sums back to host
    double *h_partial_sums = (double*)malloc(totalThreads * sizeof(double));
    CHECK_CUDA( cudaMemcpy(h_partial_sums, d_partial_sums, totalThreads * sizeof(double), cudaMemcpyDeviceToHost) );

    // Reduce the partial sums on the host
    double pi_approx = 0.0;
    for (size_t i = 0; i < totalThreads; ++i) {
        pi_approx += h_partial_sums[i];
    }

    // Print result
    printf("Approximated value of PI = %.16lf\n", pi_approx);

    // Cleanup
    free(h_partial_sums);
    CHECK_CUDA( cudaFree(d_partial_sums) );

    return 0;
}
