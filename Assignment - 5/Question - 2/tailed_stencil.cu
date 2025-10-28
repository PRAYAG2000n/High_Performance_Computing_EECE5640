#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 64 // Change the value of N as per convenience
#define BLOCK_SIZE 8

// CPU implementation for verification
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

// Tiled GPU kernel using shared memory
__global__ void gpu_stencil_tiled(float *a, const float *b) {
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int bz = blockIdx.z * BLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int x = bx + tx;
    int y = by + ty;
    int z = bz + tz;

    // Load elements into the shared memory tile
    if (x < N && y < N && z < N) {
        tile[tx + 1][ty + 1][tz + 1] = b[x * N * N + y * N + z];
        
        // Load halo elements
        if (tx == 0 && x > 0) tile[0][ty + 1][tz + 1] = b[(x - 1) * N * N + y * N + z];
        if (tx == BLOCK_SIZE - 1 && x < N - 1) tile[BLOCK_SIZE + 1][ty + 1][tz + 1] = b[(x + 1) * N * N + y * N + z];
        if (ty == 0 && y > 0) tile[tx + 1][0][tz + 1] = b[x * N * N + (y - 1) * N + z];
        if (ty == BLOCK_SIZE - 1 && y < N - 1) tile[tx + 1][BLOCK_SIZE + 1][tz + 1] = b[x * N * N + (y + 1) * N + z];
        if (tz == 0 && z > 0) tile[tx + 1][ty + 1][0] = b[x * N * N + y * N + (z - 1)];
        if (tz == BLOCK_SIZE - 1 && z < N - 1) tile[tx + 1][ty + 1][BLOCK_SIZE + 1] = b[x * N * N + y * N + (z + 1)];
    }

    __syncthreads();

    // Compute the stencil
    if (tx > 0 && tx < BLOCK_SIZE - 1 && ty > 0 && ty < BLOCK_SIZE - 1 && tz > 0 && tz < BLOCK_SIZE - 1 &&
        x > 0 && x < N - 1 && y > 0 && y < N - 1 && z > 0 && z < N - 1) {
        a[x * N * N + y * N + z] = 0.75f * (tile[tx - 1][ty + 1][tz + 1] +
                                           tile[tx + 1][ty + 1][tz + 1] +
                                           tile[tx + 1][ty - 1][tz + 1] +
                                           tile[tx + 1][ty + 1][tz - 1] +
                                           tile[tx + 1][ty + 1][tz + 1] +
                                           tile[tx + 1][ty + 1][tz + 1]);
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

    // Setup kernel dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch tiled kernel
    auto start = std::chrono::high_resolution_clock::now();
    gpu_stencil_tiled<<<gridDim, blockDim>>>(dev_a, dev_b);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Copy results back to host
    cudaMemcpy(host_a, dev_a, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the value of N
    std::cout << "Value of N: " << N << std::endl;
    // Print performance results
    std::cout << "Tiled execution time: " << duration.count() << " ms" << std::endl;

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}
