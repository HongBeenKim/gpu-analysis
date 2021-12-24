#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <limits>
#include <cmath>
#include <cuda.h>
#include "base_info.h"
#include "get_random.cu"
#include "matrix_cal.cu"

#define N (1024 * 4)


int main(){
    float gpu_elapsed_time_ms;
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

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
    cudaDeviceSynchronize();

    char c = 0;
    printf("set done\n");
    while((c = getchar()) != 'y');

    cudaEventRecord(start, cuda_stream);
    d_mm_shared_mem<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, cuda_stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("shared memory mm : %f ms\n", gpu_elapsed_time_ms);

    // free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

