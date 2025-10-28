#include <iostream>
#include <pthread.h>
#include <chrono>

struct LeibnizData {
    int terms_per_thread;
    double pi_estimate = 0.0;
};

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
        std::cerr << "Usage: " << argv[0] << " <number of threads> <number of terms>" << std::endl;
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    int total_terms = std::stoi(argv[2]);

    pthread_t threads[num_threads];
    LeibnizData lz_data[num_threads];

    int terms_per_thread = total_terms / num_threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        lz_data[i].terms_per_thread = terms_per_thread;
        pthread_create(&threads[i], nullptr, leibniz_thread, &lz_data[i]);
    }

    double total_pi_estimate = 0.0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
        total_pi_estimate += lz_data[i].pi_estimate;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    double pi_leibniz = total_pi_estimate / num_threads;

    std::cout << "Leibniz Pi = " << pi_leibniz << std::endl;
    std::cout << "Computation Time = " << elapsed.count() << " seconds" << std::endl;

    return 0;
}