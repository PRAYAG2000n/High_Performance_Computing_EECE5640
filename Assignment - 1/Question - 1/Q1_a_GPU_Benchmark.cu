
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

long loop_count;
int smCount, cudaCores;
int totalThreads;
double f_avg = 0.0; 
struct timeval t;

__global__ void FAdd(float *d_a, float *d_b, float *d_c, int totalThreads, long l_c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalThreads)
    {
        for (long j = 0; j < l_c; j++)
        {
            d_c[i] = d_c[i] + d_c[i];
            d_b[i] = d_b[i] + d_b[i];
            d_a[i] = d_a[i] + d_a[i];
        }
    }
}

double FLOPSOnce()
{
    float *fa, *fb, *fc;
    float *d_fa, *d_fb, *d_fc;
    double start, end;

    fa = (float *)malloc(totalThreads * sizeof(float));
    fb = (float *)malloc(totalThreads * sizeof(float));
    fc = (float *)malloc(totalThreads * sizeof(float));

    cudaMalloc(&d_fa, totalThreads * sizeof(float));
    cudaMalloc(&d_fb, totalThreads * sizeof(float));
    cudaMalloc(&d_fc, totalThreads * sizeof(float));

    for (int i = 0; i < totalThreads; i++) {
        fa[i] = 0.000001f;
        fb[i] = 0.000001f;
        fc[i] = 0.000001f;
    }

    cudaMemcpy(d_fa, fa, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fb, fb, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc, fc, totalThreads*sizeof(float), cudaMemcpyHostToDevice);

    gettimeofday(&t, NULL);
    start = t.tv_sec + (t.tv_usec / 1e6);

    FAdd<<< smCount, cudaCores >>>(d_fa, d_fb, d_fc, totalThreads, loop_count);
    cudaDeviceSynchronize();

    gettimeofday(&t, NULL);
    end = t.tv_sec + (t.tv_usec / 1e6);

    free(fa);
    free(fb);
    free(fc);
    cudaFree(d_fa);
    cudaFree(d_fb);
    cudaFree(d_fc);

    return (end - start);
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: %s [loop_count]\n", argv[0]);
        return 1;
    }
    loop_count = atol(argv[1]);

    smCount = 1;      
    cudaCores = 256;  
    totalThreads = smCount * cudaCores;

    printf("loop_count = %ld\n", loop_count);
    printf("smCount = %d, cudaCores = %d => totalThreads = %d\n",
           smCount, cudaCores, totalThreads);

    int NUM_RUNS = 10;
    double times[NUM_RUNS];
    double sum = 0.0;

    for (int i = 0; i < NUM_RUNS; i++)
    {
        double elapsed = FLOPSOnce();
        times[i] = elapsed;
        sum += elapsed;
    }

    printf("\nFLOPS Benchmark Times (sec), %d runs:\n", NUM_RUNS);
    for (int i = 0; i < NUM_RUNS; i++)
    {
        printf("  Run %d: %f sec\n", i+1, times[i]);
    }

    double avg_time = sum / NUM_RUNS;
    printf("Avg time: %f sec\n", avg_time);

    double total_ops = (double) loop_count * 10.0 * (double) totalThreads;
    double gflops = (total_ops / avg_time) / 1e9;
    printf("Approx. throughput: %.3f GFLOPs\n", gflops);

    return 0;
}
