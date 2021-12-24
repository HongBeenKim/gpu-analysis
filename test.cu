#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <limits>
#include <cmath>
#include <cuda.h>
#include "base_info.h"
#include "get_random.cu"
#include "matrix_cal.cu"

#define N 1024


int main(){
    float gpu_elapsed_time_ms;
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

    // make event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // device matrix malloc
    int *d_A, *d_B, *d_C, *d_D;
    int *h_C, *h_D, *h_A, *h_B, *h_asd;
    cudaMalloc((void **) &d_A, sizeof(int)*N*N);
    cudaMalloc((void **) &d_B, sizeof(int)*N*N);
    cudaMalloc((void **) &d_C, sizeof(int)*N*N);
    cudaMalloc((void **) &d_D, sizeof(int)*N*N);

    h_C = (int*)malloc(sizeof(int)*N*N);
    h_D = (int*)malloc(sizeof(int)*N*N);
    h_A = (int*)malloc(sizeof(int)*N*N);
    h_B = (int*)malloc(sizeof(int)*N*N);
    h_asd = (int*)malloc(sizeof(int)*N*N);

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int seed;
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_A, N);
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_A, d_A, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, sizeof(int)*N*N, cudaMemcpyDeviceToHost);

    // char c = 0;
    // printf("set done");
    // while((c = getchar()) != 'y');

    cudaEventRecord(start, cuda_stream);
    d_mm_shared_mem<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    

    d_mm_normal<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B, d_D, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_D, d_D, sizeof(int)*N*N, cudaMemcpyDeviceToHost);

    h_mm(h_A, h_B, h_asd, N);

    for (int i=0;i<N*N;i++) {
        if (h_C[i] != h_D[i]) printf("%d\n", i);
    }

        // for (int i=0;i<N;i++){
        //         for (int j=0;j<N;j++){
        //     printf("%d ", h_A[i*N+j]);
        //         }
        //         printf("\n");
        // }

        // for (int i=0;i<N;i++){
        //         for (int j=0;j<N;j++){
        //     printf("%d ", h_B[i*N+j]);
        //         }
        //         printf("\n");
        // }

        //                 for (int i=0;i<N;i++){
        //         for (int j=0;j<N;j++){
        //     printf("%d ", h_asd[i*N+j]);
        //         }
        //         printf("\n");
        // }

        //                 for (int i=0;i<N;i++){
        //         for (int j=0;j<N;j++){
        //     printf("%d ", h_D[i*N+j]);
        //         }
        //         printf("\n");
        // }

        
        //                 for (int i=0;i<N;i++){
        //         for (int j=0;j<N;j++){
        //     printf("%d ", h_C[i*N+j]);
        //         }
        //         printf("\n");
        // }

    // for (int i=0;i<N*N;i++){
    //     if(h_D[i] != h_asd[i*N+j]){
    //         printf("%d asd\n", i);
    //     }
        // if(h_D[i]!=h_asd[i]) printf("nor\n");
    // 

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

