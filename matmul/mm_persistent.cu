#define GRID_SIZE 32
#define BLOCK_SIZE 32

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <limits>
#include <cmath>
#include <cuda.h>
#include "get_random.cu"
#include "matrix_cal.cu"

static volatile int keepRunning = 1;

void intHandler(int dummy) {
    keepRunning = 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        exit(1);
    }
    int N;
    sscanf(argv[1], "%d", &N);

    // device matrix malloc
    int *d_A, *d_B, *d_C;
    if (cudaMalloc((void **) &d_A, sizeof(int)*N*N) != cudaSuccess) {
        printf("cudaMalloc A Failed\n");
        exit(1);
    }

    if (cudaMalloc((void **) &d_B, sizeof(int)*N*N) != cudaSuccess) {
        printf("cudaMalloc B Failed\n");
        exit(1);
    }

    if (cudaMalloc((void **) &d_C, sizeof(int)*N*N) != cudaSuccess) {
        printf("cudaMalloc C Failed\n");
        exit(1);
    }

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    /*
    int seed;
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_A, N);
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_B, N);
    */

    signal(SIGINT, intHandler);
    while (keepRunning)
        d_mm_normal<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

