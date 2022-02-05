#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <limits>
#include <cmath>
#include <cuda.h>
#include "base_info.h"
#include "get_random.cu"
#include "matrix_cal.cu"


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        exit(1);
    }
    int N;
    sscanf(argv[1], "%d", &N);

    //float gpu_elapsed_time_ms;
    float elap_time_disturb;
    cudaStream_t stream1, stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

    // make event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // device matrix malloc
    int *d_A2, *d_B2, *d_C2;
    int *d_A3, *d_B3, *d_C3;

    cudaMalloc((void **) &d_A2, sizeof(int)*N*N);
    cudaMalloc((void **) &d_B2, sizeof(int)*N*N);
    cudaMalloc((void **) &d_C2, sizeof(int)*N*N);

    cudaMalloc((void **) &d_A3, sizeof(int)*N*N);
    cudaMalloc((void **) &d_B3, sizeof(int)*N*N);
    cudaMalloc((void **) &d_C3, sizeof(int)*N*N);

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

/*
    int seed;
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_A1, N);
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_B1, N);

    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_A2, N);
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_B2, N);

    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_A3, N);
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_B3, N);
    cudaDeviceSynchronize();
*/

    // Disturbing kernel 
    d_mm_normal<<<dimGrid, dimBlock, 0, stream2>>>(d_A2, d_B2, d_C2, N);
    // Disturbed kernel 
    cudaEventRecord(start, stream1);
    d_mm_normal<<<dimGrid, dimBlock, 0, stream1>>>(d_A3, d_B3, d_C3, N);
    cudaEventRecord(stop, stream1);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elap_time_disturb, start, stop);
    printf("%f\n", elap_time_disturb);
    
    // free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);

    cudaFree(d_A3);
    cudaFree(d_B3);
    cudaFree(d_C3);
    return 0;
}

