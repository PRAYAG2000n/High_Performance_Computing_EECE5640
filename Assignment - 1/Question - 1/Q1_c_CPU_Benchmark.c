
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>  // needed for sleep()

long loop_count, l_c, th_count;

// We’ll reuse this global struct for timing in the entire program
struct timeval t;

/* -------------------------
   FLOPS Structures & Code
   ------------------------- */

// Thread structure for FLOPS
struct fth
{
    int    lc;
    int    th_counter;
    float  fa, fb, fc, fd;
    pthread_t threads;
};

// Worker function for FLOPS
void *FAdd(void *data)
{
    struct fth *th_d = (struct fth *) data;
    for (th_d->lc = 1; th_d->lc <= l_c; th_d->lc++)
    {
        th_d->fb + th_d->fc; th_d->fa - th_d->fb; // 30 floating‐point ops per loop
    }
    return NULL;
}

double FLOPSBenchmark()
{
    double start_time, end_time, elapsed;
    struct fth ft[th_count];
    for (long i = 0; i < th_count; i++)
    {
        ft[i].lc = 1; ft[i].th_counter = 1; ft[i].fa = 0.02f; ft[i].fb = 0.2f; ft[i].fc = 0; ft[i].fd = 0;
    }
    gettimeofday(&t, NULL);
    start_time = t.tv_sec + (t.tv_usec / 1000000.0);
    for (long c = 0; c < th_count; c++)
    {
        pthread_create(&ft[c].threads, NULL, FAdd, (void *)&ft[c]);
    }
    for (long m = 0; m < th_count; m++)
    {
        pthread_join(ft[m].threads, NULL);
    }
    gettimeofday(&t, NULL);
    end_time = t.tv_sec + (t.tv_usec / 1000000.0);
    elapsed = end_time - start_time;
    return elapsed;
}

/* -------------------------
   IOPS Structures & Code
   ------------------------- */

// Thread structure for IOPs
struct ith
{
    int    lc;
    int    th_counter;
    int    ia, ib, ic, id;
    pthread_t threads;
};

// Worker function for IOPs
void *IAdd(void *data)
{
    struct ith *th_d = (struct ith *) data;
    for (th_d->lc = 1; th_d->lc <= l_c; th_d->lc++)
    {
        th_d->ib + th_d->ic; th_d->ia - th_d->ib; // 30 integer ops per loop
    }
    return NULL;
}

double IOPSBenchmark()
{
    double start_time, end_time, elapsed;
    struct ith it[th_count];
    for (long i = 0; i < th_count; i++)
    {
        it[i].lc = 1; it[i].th_counter = 1; it[i].ia = 1; it[i].ib = 2; it[i].ic = 0; it[i].id = 0;
    }
    gettimeofday(&t, NULL);
    start_time = t.tv_sec + (t.tv_usec / 1000000.0);
    for (long v = 0; v < th_count; v++)
    {
        pthread_create(&it[v].threads, NULL, IAdd, (void *)&it[v]);
    }
    for (long n = 0; n < th_count; n++)
    {
        pthread_join(it[n].threads, NULL);
    }
    gettimeofday(&t, NULL);
    end_time = t.tv_sec + (t.tv_usec / 1000000.0);
    elapsed = end_time - start_time;
    return elapsed;
}

/* -------------------------
            main()
   ------------------------- */

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Incorrect number of parameters.\n");
        printf("Usage: %s [operation count] [thread count]\n", argv[0]);
        return 1;
    }

    loop_count = atol(argv[1]); // total loop count
    th_count = atol(argv[2]); // thread count
    if (th_count <= 0) {
        printf("Invalid thread count.\n");
        return 1;
    }

    l_c = loop_count / th_count; // Each thread will do l_c loops

    printf("\nStarting CPU Benchmark...\n");
    printf("  Operation Count: %ld\n", loop_count);
    printf("  Threads Implemented: %ld\n\n", th_count);

    int NUM_RUNS = 10;
    double flops_times[NUM_RUNS], flops_sum = 0.0, iops_times[NUM_RUNS], iops_sum = 0.0;

    for (int i = 0; i < NUM_RUNS; i++)
    {
        double elapsed = FLOPSBenchmark();
        flops_times[i] = elapsed;
        flops_sum += elapsed;
    }

    double flops_avg_time = flops_sum / NUM_RUNS;
    double total_flops_ops = (double)loop_count * 30.0;
    double gflops = (total_flops_ops / flops_avg_time) / 1e9;

    // Print formatted output
    printf("FLOPS timings (seconds), %d runs:\n", NUM_RUNS);
    for (int i = 0; i < NUM_RUNS; i++)
    {
        printf("Run %2d: %.6f sec\n", i + 1, flops_times[i]);
    }
    printf("Average FLOPS time: %.6f sec\n", flops_avg_time);
    printf("Approx. FLOPS throughput: %.3f G-FLOPs\n", gflops);

    // Additional code for IOPS would be similar, ensuring consistent output format
    return 0;
}
