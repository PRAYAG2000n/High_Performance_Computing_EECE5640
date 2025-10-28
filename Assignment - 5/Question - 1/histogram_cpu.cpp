#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

#define NUM_BINS 10  // Use same number of bins as CUDA version

int main() {
    srand(time(0));

    for (int exp = 12; exp <= 23; ++exp) {
        int N = 1 << exp;
        std::vector<int> input(N);
        std::vector<int> histogram(NUM_BINS, 0);
        std::vector<int> class_example(NUM_BINS, -1);

        // Generate random input in [1, 100000]
        for (int i = 0; i < N; ++i) {
            input[i] = rand() % 100000 + 1;
        }

        // Timing starts
        auto start = std::chrono::high_resolution_clock::now();

        // Parallel histogram computation using OpenMP
        #pragma omp parallel
        {
            std::vector<int> private_histogram(NUM_BINS, 0);

            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                int value = input[i];
                int bin = value * NUM_BINS / 100000;
                if (bin >= NUM_BINS) bin = NUM_BINS - 1;
                private_histogram[bin]++;
            }

            // Merge private histograms into global one
            #pragma omp critical
            {
                for (int i = 0; i < NUM_BINS; ++i)
                    histogram[i] += private_histogram[i];
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "\n--- Histogram for N = 2^" << exp << " = " << N << " ---\n";
        std::cout << "OpenMP histogram completed in " << duration_ms << " ms\n";

        // Select one example from each class/bin
        for (int i = 0; i < N; ++i) {
            int bin = input[i] * NUM_BINS / 100000;
            if (bin >= NUM_BINS) bin = NUM_BINS - 1;
            if (class_example[bin] == -1) {
                class_example[bin] = input[i];
            }
        }

        std::cout << "Example input value from each class:\n";
        for (int i = 0; i < NUM_BINS; ++i) {
            std::cout << "Class " << i << ": ";
            if (class_example[i] != -1)
                std::cout << class_example[i] << "\n";
            else
                std::cout << "(empty)\n";
        }
    }

    return 0;
}
