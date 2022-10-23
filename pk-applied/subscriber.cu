#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;


__global__ void matmul(volatile int *sharedBuf, int *B, int *C, int N, int numIter)
{
  grid_group g = this_grid();
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int gridLen = gridDim.x * blockDim.x;
  volatile int *A = sharedBuf;
  volatile int *pSyncFlag = &sharedBuf[N * N + 63];

  for (int iter = 0; iter < numIter; iter++) {
    while (*pSyncFlag % 2 == 0); 

    // Perform matrix multiplication 
    for (int i = row; i < N; i += gridLen) {
      for (int j = col; j < N; j += gridLen) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
          sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
      }
    }
    // Sync over all blocks 
    g.sync();
    if (row == 0 && col == 0)
      //atomicAdd(pSyncFlag, 1);
      *pSyncFlag += 1;
    g.sync();
  }
}


int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <port_file_name> <mat_size>\n", argv[0]);
    exit(1);
  }

  int N, num_iter = 100;
  int client_sock;
  int byte_left, byte_read;
  struct sockaddr_un serveraddr;

  dim3 dimGrid(4, 4);
  dim3 dimBlock(16, 16);

  cudaError_t err;
  cudaIpcMemHandle_t memHandle;

  int *shared_buf;
  int *h_weight, *d_weight;
  int *h_result, *d_result;
  
  char *buf;

  sscanf(argv[2], "%d", &N);

  cudaEvent_t kernelEnd;
  cudaEventCreate(&kernelEnd);

  // Allocate host & device memory 
  h_weight = (int *)malloc(sizeof(int) * N * N);
  h_result = (int *)malloc(sizeof(int) * N * N);

  if ((err = cudaMalloc((void**)&d_weight, sizeof(int) * N * N))) {
    printf("cudaMalloc() failed (%d)\n", err);
    exit(1);
  }
  if (cudaMalloc((void**)&d_result, sizeof(int) * N * N)) {
    printf("cudaMalloc() failed\n");
    exit(1);
  }

  // Identity matrix for matmul test 
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      h_weight[i * N + j] = (i == j) ? 2 : 0;
    }
  }

  cudaMemcpy(d_weight, h_weight, sizeof(int) * N * N, cudaMemcpyHostToDevice);

  if ((client_sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
    perror("socket() error: ");
    exit(1);
  }

  bzero(&serveraddr, sizeof(serveraddr));
  serveraddr.sun_family = AF_UNIX;
  strcpy(serveraddr.sun_path, argv[1]);

  if (connect(client_sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) < 0) {
    perror("connect() error: ");
    exit(1);
  }

  // receive GPU mem handle 
  buf = (char *)&memHandle;
  byte_left = sizeof(memHandle);
  while (byte_left > 0) {
    byte_read = read(client_sock, buf, byte_left);
    byte_left -= byte_read;
    buf += byte_read;
  }

  err = cudaIpcOpenMemHandle(
    (void**)&shared_buf, memHandle, cudaIpcMemLazyEnablePeerAccess
  );
  if (err) {
    printf("cudaIpcOpenMemHandle() failed (%d)\n", err);
    exit(1);
  }

  // Launch kernel 
  void *kernelArgs[] = { &shared_buf, &d_weight, &d_result, &N, &num_iter };
  cudaLaunchCooperativeKernel((void*)matmul, dimGrid, dimBlock, kernelArgs);
  cudaEventRecord(kernelEnd, 0);
  cudaEventSynchronize(kernelEnd);

  // cleanup 
  close(client_sock);
  cudaIpcCloseMemHandle(shared_buf);

  cudaFree(d_weight);
  cudaFree(d_result);

  free(h_weight);
  free(h_result);

  cudaEventDestroy(kernelEnd);

  return 0;
}

