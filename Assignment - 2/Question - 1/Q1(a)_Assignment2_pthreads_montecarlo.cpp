#include <iostream>
#include <pthread.h>
#include <random>
#include <chrono>
#include <cmath>

struct MonteCarloData {
    int darts_per_thread;
    int count_inside_circle = 0;
};

struct LeibnizData {
    int terms_per_thread;
    double pi_estimate = 0.0;
};

void* monte_carlo_thread(void* arg) {
    MonteCarloData* data = static_cast<MonteCarloData*>(arg);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int count = 0;
    for (int i = 0; i < data->darts_per_thread; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        if (x * x + y * y <= 1) {
            count++;
        }
    }
    data->count_inside_circle = count;
    return nullptr;
}

void* leibniz_thread(void* arg) {
    LeibnizData* data = static_cast<LeibnizData*>(arg);
    double pi_estimate = 0.0;
    for (int i = 0; i < data->terms_per_thread; ++i) {
        pi_estimate += (i % 2 == 0 ? 1 : -1) / (2.0 * i + 1);
    }
    data->pi_estimate = pi_estimate * 4;
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number of threads> <number of darts>" << std::endl;
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    int total_darts = std::stoi(argv[2]);
    int total_terms = total_darts; // Assuming terms equal to darts for simplicity

    pthread_t threads[2 * num_threads];
    MonteCarloData mc_data[num_threads];
    LeibnizData lz_data[num_threads];

    int darts_per_thread = total_darts / num_threads;
    int terms_per_thread = total_terms / num_threads;

    // Start timing for Monte Carlo
    auto mc_start_time = std::chrono::high_resolution_clock::now();

    // Launch Monte Carlo threads
    for (int i = 0; i < num_threads; ++i) {
        mc_data[i].darts_per_thread = darts_per_thread;
        pthread_create(&threads[i], nullptr, monte_carlo_thread, &mc_data[i]);
    }

    int total_inside_circle = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
        total_inside_circle += mc_data[i].count_inside_circle;
    }

    auto mc_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> mc_elapsed = mc_end_time - mc_start_time;

    // Start timing for Leibniz
    auto lz_start_time = std::chrono::high_resolution_clock::now();

    // Launch Leibniz threads
    for (int i = 0; i < num_threads; ++i) {
        lz_data[i].terms_per_thread = terms_per_thread;
        pthread_create(&threads[num_threads + i], nullptr, leibniz_thread, &lz_data[i]);
    }

    double total_pi_estimate = 0.0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[num_threads + i], nullptr);
        total_pi_estimate += lz_data[i].pi_estimate;
    }

    auto lz_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> lz_elapsed = lz_end_time - lz_start_time;

    double pi_monte_carlo = 4.0 * total_inside_circle / total_darts;
    double pi_leibniz = total_pi_estimate / num_threads;

    std::cout << "Monte Carlo Pi = " << pi_monte_carlo << std::endl;
    std::cout << "Leibniz Pi = " << pi_leibniz << std::endl;
    std::cout << "Monte Carlo Time = " << mc_elapsed.count() << " seconds" << std::endl;
    std::cout << "Leibniz Time = " << lz_elapsed.count() << " seconds" << std::endl;

    return 0;
}
