#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdlib>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // -----------------------------------------------------------------------
    // 1) Parse command-line arguments for number of bins
    // -----------------------------------------------------------------------
    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <num_bins>\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_bins = std::atoi(argv[1]);
    if (num_bins <= 0) {
        if (rank == 0) {
            std::cerr << "num_bins must be > 0\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // -----------------------------------------------------------------------
    // 2) Set up constants
    // -----------------------------------------------------------------------
    const long long TOTAL_VALUES = 8000000LL;  // 8 million integers
    const int DATA_MIN = 1;
    const int DATA_MAX = 100000;
    long long local_count = TOTAL_VALUES / size;  
    // (Optionally handle leftover if TOTAL_VALUES is not perfectly divisible.)

    // -----------------------------------------------------------------------
    // 3) Prepare local data
    // -----------------------------------------------------------------------
    // Each process will generate local_count random integers.
    // We'll seed each process differently using rank + time.
    std::vector<int> local_data(local_count);
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr) + rank));
    std::uniform_int_distribution<int> dist(DATA_MIN, DATA_MAX);

    // -----------------------------------------------------------------------
    // 4) Start overall timer
    // -----------------------------------------------------------------------
    double start_time = MPI_Wtime();

    // -----------------------------------------------------------------------
    // 5) Generate local random data
    // -----------------------------------------------------------------------
    for (long long i = 0; i < local_count; ++i) {
        local_data[i] = dist(rng);
    }

    // -----------------------------------------------------------------------
    // 6) Create local histogram
    // -----------------------------------------------------------------------
    std::vector<long long> local_hist(num_bins, 0LL);
    double bin_width = static_cast<double>(DATA_MAX - DATA_MIN + 1) / num_bins;

    for (auto value : local_data) {
        int bin_index = static_cast<int>((value - DATA_MIN) / bin_width);
        // Guard against floating point edge case
        if (bin_index >= num_bins) {
            bin_index = num_bins - 1;
        }
        local_hist[bin_index]++;
    }

    // -----------------------------------------------------------------------
    // 7) Reduce local histograms into global histogram
    //    We'll store the result on rank 0
    // -----------------------------------------------------------------------
    std::vector<long long> global_hist;
    if (rank == 0) {
        global_hist.resize(num_bins, 0LL);
    }

    // Notice that MPI_LONG_LONG might differ by system,
    // but typically we can use MPI_LONG_LONG or MPI_LONG.
    MPI_Reduce(local_hist.data(), 
               (rank == 0 ? global_hist.data() : nullptr), 
               num_bins, 
               MPI_LONG_LONG, 
               MPI_SUM, 
               0, 
               MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // 8) Stop overall timer
    // -----------------------------------------------------------------------
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // -----------------------------------------------------------------------
    // 9) Output results (only on rank 0)
    // -----------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "Final Histogram (" << num_bins << " bins):\n";
        long long total_sum = 0;
        for (int i = 0; i < num_bins; ++i) {
            std::cout << "Bin " << i << ": " << global_hist[i] << "\n";
            total_sum += global_hist[i];
        }
        std::cout << "Total values tallied: " << total_sum 
                  << " (expected ~" << TOTAL_VALUES << ")\n";
        std::cout << "Time elapsed: " << elapsed << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
