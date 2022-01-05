#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <limits>
#include <cmath>
#include <cuda.h>
#include "base_info.h"
#include "get_random.cu"
#include "matrix_cal.cu"

// #define N 64 * 20

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <matrix_size> <print_time>\n", argv[0]);
        exit(1);
    }
    int N, print;
    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &print);

    //float gpu_elapsed_time_ms;
    float elap_time;
    cudaStream_t stream1;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

    // make event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // device matrix malloc
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, sizeof(int)*N*N);
    cudaMalloc((void **) &d_B, sizeof(int)*N*N);
    cudaMalloc((void **) &d_C, sizeof(int)*N*N);

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int seed;
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_A, N);
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_B, N);

    // record matmul - single kernel 
    cudaEventRecord(start, stream1);
    d_mm_normal<<<dimGrid, dimBlock, 0, stream1>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, stream1);
    cudaEventSynchronize(stop);

    // calculate elapsed time
    cudaEventElapsedTime(&elap_time, start, stop);
    if (print)
        printf("%f\n", elap_time);
 
    // free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

