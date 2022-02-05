#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <random>

#define BLOCKS 4
#define THREADS 256

#define COMPUTE(a, b, c, d, e) (a = b * b / c - b + d * c - d)

__global__ void gpu_compute(char *arr)
{
    int i = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a, b, c, d, e;
    while (i < 10000){
        __syncthreads();
        a = arr[idx];
        b = arr[idx + 1];
        c = arr[idx + 2];
        d = arr[idx + 3];
        e = arr[idx + 4];
        __syncthreads();

        if (i & 1) {
            for (int j = 0; j < 1000; j++) {
                COMPUTE(a, b, c, d, e);
                COMPUTE(b, c, d, e, a);
                COMPUTE(c, d, e, a, b);
                COMPUTE(d, e, a, b, c);
                COMPUTE(e, a, b, c, d);
                COMPUTE(a, b, c, d, e);
                COMPUTE(b, c, d, e, a);
                COMPUTE(c, d, e, a, b);
                COMPUTE(d, e, a, b, c);
                COMPUTE(e, a, b, c, d);
                COMPUTE(a, b, c, d, e);
                COMPUTE(b, c, d, e, a);
                COMPUTE(c, d, e, a, b);
                COMPUTE(d, e, a, b, c);
                COMPUTE(e, a, b, c, d);
                COMPUTE(a, b, c, d, e);
                COMPUTE(b, c, d, e, a);
                COMPUTE(c, d, e, a, b);
                COMPUTE(d, e, a, b, c);
                COMPUTE(e, a, b, c, d);
            }
        }
        else {
            for (int j = 0; j < 1000; j++) {
                COMPUTE(e, d, c, b, a);
                COMPUTE(d, c, b, a, e);
                COMPUTE(c, b, a, e, d);
                COMPUTE(b, a, e, d, c);
                COMPUTE(a, e, d, c, b);
                COMPUTE(e, d, c, b, a);
                COMPUTE(d, c, b, a, e);
                COMPUTE(c, b, a, e, d);
                COMPUTE(b, a, e, d, c);
                COMPUTE(a, e, d, c, b);
                COMPUTE(e, d, c, b, a);
                COMPUTE(d, c, b, a, e);
                COMPUTE(c, b, a, e, d);
                COMPUTE(b, a, e, d, c);
                COMPUTE(a, e, d, c, b);
                COMPUTE(e, d, c, b, a);
                COMPUTE(d, c, b, a, e);
                COMPUTE(c, b, a, e, d);
                COMPUTE(b, a, e, d, c);
                COMPUTE(a, e, d, c, b);
            }
        }
        __syncthreads();
        arr[idx] = e;
        arr[idx + 1] = d;
        arr[idx + 2] = c;
        arr[idx + 3] = b;
        arr[idx + 4] = a;
        i++;
	}
} 

int main(void) {
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 32);

	char *arr;
	char *d_arr;
	uint64_t size = BLOCKS * THREADS + 5;
    float gpu_elapsed_time_ms;

	arr = (char *)malloc(size);
    for (int i=0; i<size; i++) {
        arr[i] = dis(gen);
    }

	cudaMalloc((void **)&d_arr, size);
	cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // make event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    char c = 0;
    printf("set done\n");
    while ((c = getchar()) != 'y');

    // record matrix multiple
    cudaEventRecord(start, cuda_stream);
	gpu_compute<<< BLOCKS, THREADS, 0, cuda_stream >>>(d_arr);
	cudaDeviceSynchronize();

    cudaEventRecord(stop, cuda_stream);
    cudaEventSynchronize(stop);

    // calculate elapsed time
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("high_core: %f ms\n", gpu_elapsed_time_ms);

	// clean
	free(arr);
	cudaFree(d_arr); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	return 0;
}

