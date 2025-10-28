#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

// Global struct for timing
struct timeval t;

// Struct for FLOPS threads
typedef struct {
    int    lc;
    int    th_counter;
    float  fa, fb, fc, fd;
    pthread_t threads;
} fth;

// Worker function for FLOPS
void *FAdd(void *data)
{
    fth *th_d = (fth *) data;
    for (th_d->lc = 1; th_d->lc <= th_d->th_counter; th_d->lc++)
    {
        // Simulated operations (not stored or used)
        th_d->fb + th_d->fc;
        th_d->fa - th_d->fb;
        th_d->fa + th_d->fd;
        th_d->fa + th_d->fb;
        th_d->fb + th_d->fc;
        th_d->fa - th_d->fb;
        th_d->fa + th_d->fd;
        th_d->fa + th_d->fb;
        th_d->fb + th_d->fc;
        th_d->fa - th_d->fb;

        th_d->fb + th_d->fc;
        th_d->fa - th_d->fb;
        th_d->fa + th_d->fd;
        th_d->fa + th_d->fb;
        th_d->fb + th_d->fc;
        th_d->fa - th_d->fb;
        th_d->fa + th_d->fd;
        th_d->fa + th_d->fb;
        th_d->fb + th_d->fc;
        th_d->fa - th_d->fb;

        th_d->fb + th_d->fc;
        th_d->fa - th_d->fb;
        th_d->fa + th_d->fd;
        th_d->fa + th_d->fb;
        th_d->fb + th_d->fc;
        th_d->fa - th_d->fb;
        th_d->fa + th_d->fd;
        th_d->fa + th_d->fb;
        th_d->fb + th_d->fc;
        th_d->fa - th_d->fb;
        // Repeat operations to simulate computational load
    }
    return NULL;
}

// FLOPS Benchmark Function
double FLOPSBenchmark(long loop_count, long th_count)
{
    fth *ft = malloc(sizeof(fth) * th_count);
    double start_time, end_time, elapsed;

    // Initialize data for each thread
    for (long i = 0; i < th_count; i++)
    {
        ft[i].lc = loop_count / th_count;
        ft[i].th_counter = ft[i].lc;
        ft[i].fa = 0.02f;
        ft[i].fb = 0.2f;
        ft[i].fc = 0;
        ft[i].fd = 0;
        pthread_create(&ft[i].threads, NULL, FAdd, &ft[i]);
    }

    // Timing
    gettimeofday(&t, NULL);
    start_time = t.tv_sec + (t.tv_usec / 1000000.0);

    // Join threads
    for (long m = 0; m < th_count; m++)
    {
        pthread_join(ft[m].threads, NULL);
    }

    gettimeofday(&t, NULL);
    end_time = t.tv_sec + (t.tv_usec / 1000000.0);
    elapsed = end_time - start_time;

    free(ft);
    return elapsed;
}

// Main function
int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Incorrect number of parameters.\n");
        printf("Usage: %s [operation count] [thread count]\n", argv[0]);
        return 1;
    }

    long loop_count = atol(argv[1]);  // total loop count
    long th_count = atol(argv[2]);    // thread count

    if (th_count <= 0) {
        printf("Invalid thread count.\n");
        return 1;
    }

    const int NUM_RUNS = 10;
    double times[NUM_RUNS];
    double sum = 0.0;

    printf("CPU Benchmark timings (seconds), %d runs:\n", NUM_RUNS);
    for (int i = 0; i < NUM_RUNS; i++) {
        times[i] = FLOPSBenchmark(loop_count, th_count);
        sum += times[i];
        printf("Run %d: %.6f sec\n", i + 1, times[i]);
    }

    double avg_time = sum / NUM_RUNS;
    printf("Average CPU Benchmark FLOPS time: %.6f sec\n", avg_time);

    return 0;
}


