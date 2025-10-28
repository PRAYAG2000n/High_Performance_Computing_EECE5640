#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 512
#define LOOPS 10
#define BLOCK_SIZE 32  // Adjust as needed (8, 16, 32, etc.)

static double a[N][N], b[N][N], c[N][N], bT[N][N];  // bT for transposed B

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

int main()
{
    int i, j, k, ii, jj, kk, l;
    double start, finish, total;

    // ---------------------------------------------------
    // 1) DENSE INITIALIZATION + TRANSPOSE
    // ---------------------------------------------------
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++){
            a[i][j] = (double)(i + j);
            b[i][j] = (double)(i - j);
        }
    }

    // Create the transpose of b to improve cache locality
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++){
            bT[j][i] = b[i][j];
        }
    }

    // ---------------------------------------------------
    // 2) DENSE MULTIPLICATION WITH BLOCKING
    // ---------------------------------------------------
    printf("starting dense matrix multiply with blocking...\n");
    start = CLOCK();

    for(l = 0; l < LOOPS; l++) {
        // Zero out c once per LOOPS iteration
        #pragma omp parallel for private(i,j)
        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++) {
                c[i][j] = 0.0;
            }
        }

        // Tiled multiplication
        #pragma omp parallel for private(ii,jj,kk,i,j,k)
        for(ii = 0; ii < N; ii += BLOCK_SIZE) {
            for(jj = 0; jj < N; jj += BLOCK_SIZE) {
                for(kk = 0; kk < N; kk += BLOCK_SIZE) {
                    // Now multiply the sub-blocks
                    for(i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                        for(j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                            double sum = c[i][j];
                            // Instead of b[k][j], we use bT[j][k] for better locality
                            for(k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                                sum += a[i][k] * bT[j][k];
                            }
                            c[i][j] = sum;
                        }
                    }
                }
            }
        }
    }

    finish = CLOCK();
    total = finish - start;
    printf("a result %g \n", c[7][8]);
    printf("Total time for dense matmul (with tiling + transpose) = %f ms\n", total);

    // ---------------------------------------------------
    // 3) SPARSE MULTIPLICATION
    // (Optionally also compress B or tile in the sparse case)
    // ---------------------------------------------------
    // The rest is similar to the earlier example: build compressed A (and possibly B) 
    // and only iterate over the nonzero entries.

    // We'll show the typical approach compressing A only. 
    // For further optimization, consider also compressing B or even 
    // exploring advanced reordering / blocking.

    int num_zeros = 0;
    // Re-initialize 'a' and 'b' for sparse
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            if ((i < j) && (i % 2 > 0)) {
                a[i][j] = (double)(i + j);
                b[i][j] = (double)(i - j);
            } else {
                num_zeros++;
                a[i][j] = 0.0;
                b[i][j] = 0.0;
            }
        }
    }

    // Build the transpose of b (optional if you want to attempt better locality)
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++){
            bT[j][i] = b[i][j];
        }
    }

    // Build compressed row structure for A
    typedef struct {
        int count;
        int *colIndex;
        double *val;
    } SparseRow;

    SparseRow *Acomp = (SparseRow*) malloc(N * sizeof(SparseRow));
    for(i = 0; i < N; i++){
        // count how many nonzeros in row i
        int row_count = 0;
        for(j = 0; j < N; j++){
            if(a[i][j] != 0.0) {
                row_count++;
            }
        }
        Acomp[i].count = row_count;
        Acomp[i].colIndex = (int*) malloc(row_count * sizeof(int));
        Acomp[i].val = (double*) malloc(row_count * sizeof(double));

        // store them
        int idx = 0;
        for(j = 0; j < N; j++){
            if(a[i][j] != 0.0) {
                Acomp[i].colIndex[idx] = j;
                Acomp[i].val[idx] = a[i][j];
                idx++;
            }
        }
    }

    // Multiply with the sparse A
    printf("starting sparse matrix multiply...\n");
    start = CLOCK();
    for(l = 0; l < LOOPS; l++) {
        #pragma omp parallel for private(i,j,k)
        for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
                double sum = 0.0;
                // iterate over the nonzeros in row i
                for(k = 0; k < Acomp[i].count; k++) {
                    int colIdx = Acomp[i].colIndex[k];
                    sum += Acomp[i].val[k] * bT[j][colIdx]; 
                    // we can use bT for better locality: b[k][j] => bT[j][k]
                }
                c[i][j] = sum;
            }
        }
    }
    finish = CLOCK();
    total = finish - start;
    printf("A result %g \n", c[7][8]);
    printf("Total time for sparse matmul = %f ms\n", total);
    printf("Sparsity of matrices = %f\n", (float)num_zeros/(float)(N*N));

    // Cleanup
    for(i = 0; i < N; i++){
        free(Acomp[i].colIndex);
        free(Acomp[i].val);
    }
    free(Acomp);

    return 0;
}
