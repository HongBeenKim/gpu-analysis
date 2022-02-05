__global__ void d_mm_normal(int *A, int *B, int*C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = col; i < N; i += BLOCK_SIZE * GRID_SIZE) {
        for (int j = row; j < N; j += BLOCK_SIZE * GRID_SIZE) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[j * N + k] * B[k * N + i];
            }
            C[j * N + i] = sum;
        }
    }
}

__global__ void disturb(int *A, int *B, int *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int n = 0; n < 10; n++) {
        for (int i = col; i < N; i += BLOCK_SIZE * GRID_SIZE) {
            for (int j = row; j < N; j += BLOCK_SIZE * GRID_SIZE) {
                int sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += A[j * N + k] * B[k * N + i];
                }
                C[j * N + i] = sum;
            }
        }
    }
}


__global__ void d_mm_shared_mem(int *A, int *B, int *C, int N)
{
    __shared__ int aTile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int bTile[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int p, q;

    for (int i = col; i < N; i+=BLOCK_SIZE*GRID_SIZE) {
        for (int j = row; j < N; j+=BLOCK_SIZE*GRID_SIZE) {
            int sum = 0;
            for (int tileN = 0; tileN < (N-1) / BLOCK_SIZE + 1; tileN++) {
                p = tileN * BLOCK_SIZE + threadIdx.x;
                q = tileN * BLOCK_SIZE + threadIdx.y;
                aTile[threadIdx.y][threadIdx.x] = A[j * N + p];
                bTile[threadIdx.y][threadIdx.x] = B[q * N + i];
                __syncthreads();
                
                for(int k = 0; k < BLOCK_SIZE; k++) {
                    sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
                }
            }
            C[j * N + i] = sum;
        }
    }
}


void h_mm(int *A, int *B, int *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

