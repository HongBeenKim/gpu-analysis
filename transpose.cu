#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <limits>
#include <cmath>
#include <cuda.h>
#include "base_info.h"
#include "get_random.cu"

#define N (1024)

__global__ void d_matrix_transpose(int *A, int *B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (1){
        for (int i = col; i<N; i+=BLOCK_SIZE*GRID_SIZE){
            for(int j = row; j<N; j+=BLOCK_SIZE*GRID_SIZE){
                A[i * N + j] = B[j * N + i];
            }
        }
    }
}

int main(){
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, INT_MAX);

    int *d_A, *d_B;
    cudaMalloc((void **) &d_A, sizeof(int)*N*N);
    cudaMalloc((void **) &d_B, sizeof(int)*N*N);

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int seed;
    seed = dis(gen);
    d_rand_matrix<<<1, 1>>>(seed, d_A, N);
    seed = dis(gen);
    d_rand_matrix<<<1, 1>>>(seed, d_B, N);
    cudaDeviceSynchronize();

    char c = 0;
    printf("set done\n");
    while((c = getchar()) != 'y');

    d_matrix_transpose<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B);
    cudaDeviceSynchronize();
    
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
