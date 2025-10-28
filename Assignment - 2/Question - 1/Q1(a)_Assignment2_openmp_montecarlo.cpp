#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number of threads> <number of darts>" << std::endl;
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    int total_darts = std::stoi(argv[2]);
    int darts_per_thread = total_darts / num_threads;
    int total_inside_circle = 0;

    omp_set_num_threads(num_threads);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel reduction(+:total_inside_circle)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        int count = 0;
        #pragma omp for
        for (int i = 0; i < total_darts; ++i) {
            double x = dis(gen);
            double y = dis(gen);
            if (x * x + y * y <= 1) {
                count++;
            }
        }
        total_inside_circle += count;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    double pi = 4.0 * total_inside_circle / total_darts;

    std::cout << "Monte Carlo Pi = " << pi << std::endl;
    std::cout << "Time = " << elapsed.count() << " seconds" << std::endl;

    return 0;
}