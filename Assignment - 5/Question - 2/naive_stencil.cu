#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 96 // Change the value of N as per convenience
#define BLOCK_SIZE 8

// CPU implementation (for verification)
void cpu_stencil(float a[N][N][N], float b[N][N][N]) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            for (int k = 1; k < N - 1; k++) {
                a[i][j][k] = 0.75f * (b[i - 1][j][k] + b[i + 1][j][k] +
                                       b[i][j - 1][k] + b[i][j + 1][k] +
                                       b[i][j][k - 1] + b[i][j][k + 1]);
            }
        }
    }
}

// Non-tiled GPU kernel (Native)
__global__ void gpu_stencil_native(float *a, const float *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < N - 1 && j < N - 1 && k < N - 1) {
        a[i * N * N + j * N + k] = 0.75f * (b[(i - 1) * N * N + j * N + k] + b[(i + 1) * N * N + j * N + k] +
                                           b[i * N * N + (j - 1) * N + k] + b[i * N * N + (j + 1) * N + k] +
                                           b[i * N * N + j * N + (k - 1)] + b[i * N * N + j * N + (k + 1)]);
    }
}

int main() {
    float host_a[N][N][N], host_b[N][N][N];
    float *dev_a, *dev_b;

    // Initialize host_b
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                host_b[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }

    // Allocate device memory
    cudaMalloc(&dev_a, N * N * N * sizeof(float));
    cudaMalloc(&dev_b, N * N * N * sizeof(float));

    // Copy host_b to device_b
    cudaMemcpy(dev_b, host_b, N * N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Execute non-tiled kernel
    auto start = std::chrono::high_resolution_clock::now();
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 2) / BLOCK_SIZE, (N + BLOCK_SIZE - 2) / BLOCK_SIZE, (N + BLOCK_SIZE - 2) / BLOCK_SIZE);
    gpu_stencil_native<<<gridDim, blockDim>>>(dev_a, dev_b);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Copy results back to host
    cudaMemcpy(host_a, dev_a, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the value of N and performance results
    std::cout << "N Value: " << N << std::endl;
    std::cout << "Native (non-tiled) execution time: " << duration.count() << " ms" << std::endl;

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}
