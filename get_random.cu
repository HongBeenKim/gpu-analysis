#include <curand_kernel.h>
#include <random>

__device__ int get_rand_num(curandState_t *state, int A, int B){
    float rand_int = curand_uniform(state);
    rand_int = rand_int * (B - A) + A;
    return rand_int;
}

__global__ void d_rand_matrix(int seed, int* result, int N){
    curandState_t state;
    curand_init(seed, 0, 0, &state);

	for(int i = 0; i < N*N; i++){
		result[i] = get_rand_num(&state, 0, 1024); 
	}
}

int* h_rand_matrix(int *matrix, int N){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1024);
    for(int i = 0; i < N*N; i++){
        matrix[i] = dis(gen);   
    }
    return matrix;
}

int get_seed(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, INT_MAX);
    int seed = dis(gen);
    return seed;
}