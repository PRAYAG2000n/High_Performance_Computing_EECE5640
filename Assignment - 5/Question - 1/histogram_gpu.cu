#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

#define NUM_BINS 10 

__global__ void histogram_kernel(int* input, int* histogram, int N, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int value = input[idx];
        int bin = value * num_bins / 100000;
        if (bin >= num_bins) bin = num_bins - 1;
        atomicAdd(&histogram[bin], 1);
    }
}

int main() {
    srand(time(0));

    for (int exp = 12; exp <= 23; ++exp) {
        int N = 1 << exp;
        std::cout << "\n--- Histogram for N = 2^" << exp << " = " << N << " ---\n";

        std::vector<int> input_host(N);
        std::vector<int> histogram_host(NUM_BINS, 0);
        std::vector<int> class_example(NUM_BINS, -1);

        // Generate random input in [1, 100000]
        for (int i = 0; i < N; ++i) {
            input_host[i] = rand() % 100000 + 1;
        }

        // Allocate device memory
        int* input_dev = nullptr;
        int* histogram_dev = nullptr;
        cudaMalloc(&input_dev, N * sizeof(int));
        cudaMalloc(&histogram_dev, NUM_BINS * sizeof(int));

        cudaMemcpy(input_dev, input_host.data(), N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(histogram_dev, 0, NUM_BINS * sizeof(int));

        // Start GPU timing (excluding printing)
        auto start = std::chrono::high_resolution_clock::now();

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        histogram_kernel<<<gridSize, blockSize>>>(input_dev, histogram_dev, N, NUM_BINS);
        cudaDeviceSynchronize();

        cudaMemcpy(histogram_host.data(), histogram_dev, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "CUDA histogram completed in " << elapsed_ms << " ms\n";

        // Host: select one example per class
        for (int i = 0; i < N; ++i) {
            int bin = input_host[i] * NUM_BINS / 100000;
            if (bin >= NUM_BINS) bin = NUM_BINS - 1;
            if (class_example[bin] == -1) {
                class_example[bin] = input_host[i];
            }
        }

        // Print one input element from each class (ascending)
        std::cout << "Example input value from each class:\n";
        for (int i = 0; i < NUM_BINS; ++i) {
            std::cout << "Class " << i << ": ";
            if (class_example[i] != -1)
                std::cout << class_example[i] << "\n";
            else
                std::cout << "(empty)\n";
        }

        // Cleanup for current iteration
        cudaFree(input_dev);
        cudaFree(histogram_dev);
    }

    return 0;
}
