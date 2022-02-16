#define GRID_SIZE 2
#define BLOCK_SIZE 32

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <limits>
#include <cmath>
#include <cuda.h>
#include "get_random.cu"
#include "matrix_cal.cu"


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
    // cudaStream_t stream1;
    // cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

    // make event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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

    // record matmul - single kernel 
    cudaEventRecord(start, 0);
    disturb<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
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

