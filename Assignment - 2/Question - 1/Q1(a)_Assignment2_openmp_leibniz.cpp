#include <iostream>
#include <chrono>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number of threads> <number of terms>" << std::endl;
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    int total_terms = std::stoi(argv[2]);
    int terms_per_thread = total_terms / num_threads;
    double total_pi_estimate = 0.0;

    omp_set_num_threads(num_threads);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:total_pi_estimate)
    for (int i = 0; i < num_threads; ++i) {
        double pi_estimate = 0.0;
        for (int j = 0; j < terms_per_thread; ++j) {
            pi_estimate += (j % 2 == 0 ? 1 : -1) / (2.0 * j + 1);
        }
        total_pi_estimate += pi_estimate;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    double pi = total_pi_estimate * 4 / num_threads;

    std::cout << "Leibniz Pi = " << pi << std::endl;
    std::cout << "Time = " << elapsed.count() << " seconds" << std::endl;

    return 0;
}