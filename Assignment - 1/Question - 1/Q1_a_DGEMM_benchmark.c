
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
#include <malloc.h>

typedef struct algo_t {
    const char *type;
    void (*join)(const int *, const int *, int *, int, int, int, int);
    struct algo_t *pNext;
} algo_t;

static void flops_benchmark(const int *src1, const int *src2,
                            int *output, int w1, int h1, int w2, int h2)
{
    int total = w1 * h1;
    for (int i = 0; i < total; i++) {
        output[i] = src1[i] + src2[i];
    }
}

static algo_t flopsAlgo = {
    "flops",
    flops_benchmark,
    NULL
};

algo_t *list = &flopsAlgo;

static long diff_in_us(struct timespec t1, struct timespec t2)
{
    struct timespec diff;
    if (t2.tv_nsec - t1.tv_nsec < 0) {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec - 1;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec + 1000000000;
    } else {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }
    return (long)(diff.tv_sec * 1000000.0 + diff.tv_nsec / 1000.0);
}

#define TEST_W 1024
#define TEST_H 1024

int main(void)
{
    int *src1    = (int *)memalign(32, sizeof(int) * TEST_W * TEST_H);
    int *src2    = (int *)memalign(32, sizeof(int) * TEST_W * TEST_H);
    int *dst     = (int *)memalign(32, sizeof(int) * TEST_W * TEST_H);
    int *correct = (int *)memalign(32, sizeof(int) * TEST_W * TEST_H);

    srand((unsigned)time(NULL));
    for (int i = 0; i < TEST_H; ++i) {
        for (int j = 0; j < TEST_W; ++j) {
            src1[i * TEST_W + j] = rand();
            src2[i * TEST_W + j] = rand();
        }
    }

    struct timespec start, end;
    const int num_runs = 10;

    for (algo_t *tmp = list; tmp != NULL; tmp = tmp->pNext) {
        double times_s[num_runs];

        for (int run = 0; run < num_runs; run++) {
            clock_gettime(CLOCK_REALTIME, &start);
            tmp->join(src1, src2, dst, TEST_W, TEST_H, TEST_W, TEST_H);
            clock_gettime(CLOCK_REALTIME, &end);

            double elapsed_s = diff_in_us(start, end) / 1000000.0;
            times_s[run] = elapsed_s;
        }

        printf("===== %s FLOPS Benchmark =====\n", tmp->type);
        double total_s = 0.0;
        for (int run = 0; run < num_runs; run++) {
            total_s += times_s[run];
            printf("Run %d: %.6f s\n", run + 1, times_s[run]);
        }
        printf("Average time: %.6f s\n\n", (total_s / num_runs));
    }

    free(src1);
    free(src2);
    free(dst);
    free(correct);

    return 0;
}
