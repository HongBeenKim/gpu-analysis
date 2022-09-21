__global__ void fill (int *base, int size, int val) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size)
    base[idx] = val;
}

__global__ void add (int *A, int *B, int *C, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] + B[idx];

    for (int i = 0; i < 10000; i++)
      A[idx] += 1;
  }
}

__global__ void check (int *base, int size, int val) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size && base[idx] != val)
    printf("wrong! %d\n", base[idx]); 
}

