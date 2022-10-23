#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;


__global__ void generateMat(volatile int *sharedBuf, int N, int num_iter) 
{
  grid_group g = this_grid();
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int gridLen = gridDim.x * blockDim.x;
  volatile int *mat = sharedBuf;
  volatile int *pSyncFlag = &sharedBuf[N * N + 63];

  for (int iter = 0; iter < num_iter; iter++) {
    while (*pSyncFlag % 2 == 1); 

    // Perform matrix generation 
    int seed = gridLen * row + col + iter;
    curandState_t state;
    curand_init(seed, 0, 0, &state);

    float rand_int = curand_uniform(&state);
    rand_int = rand_int * 20;

    for (int i = row; i < N; i += gridLen) {
      for (int j = col; j < N; j += gridLen) {
        mat[i * N + j] = (int)rand_int;
      }
    }
    // Sync over all blocks 
    g.sync();
    if (row == 0 && col == 0) {
      *pSyncFlag += 1;
    }
    g.sync();
  }
}


int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <port_file_name> <mat_size>\n", argv[0]);
    exit(1);
  }

  int N, byte_left, byte_sent;
  int num_subscribers = 1;
  int num_iter = 100;
  int server_sock, client_sock;
  struct sockaddr_un serveraddr, clientaddr;
  socklen_t client_len = sizeof(clientaddr);
  char *buf;

  cudaError_t err;
  cudaIpcMemHandle_t memHandle;

  dim3 dimGrid(4, 4);
  dim3 dimBlock(16, 16);

  struct timeval start, end, gap;

  int *shared_buf;

  sscanf(argv[2], "%d", &N);

  cudaEvent_t kernelEnd;
  err = cudaEventCreate(&kernelEnd);
  if (err) {
    printf("cudaEventCreate failed (%d)\n", err);
    exit(1);
  }

  if (cudaMalloc((void**)&shared_buf, sizeof(int) * (N * N + 64))) {
    printf("cudaMalloc() failed\n");
    exit(1);
  }

  if ((err = cudaIpcGetMemHandle(&memHandle, shared_buf))) {
    printf("cudaIpcGetMemHandle() failed (%d)\n", err);
    exit(1);
  }

  if (access(argv[1], F_OK) == 0) {
    unlink(argv[1]);
  }

  if ((server_sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
    perror("socket() error: ");
    exit(1);
  }

  bzero(&serveraddr, sizeof(serveraddr));
  serveraddr.sun_family = AF_UNIX;
  strcpy(serveraddr.sun_path, argv[1]);

  if (bind(server_sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) < 0) {
    perror("bind() error: ");
    exit(1);
  }

  if (listen(server_sock, num_subscribers) < 0) {
    perror("listen() error: ");
    exit(1);
  }

  client_sock = accept(server_sock, (struct sockaddr *)&clientaddr, &client_len);

  // send GPU mem handle 
  buf = (char *)&memHandle;
  byte_left = sizeof(memHandle);
  while (byte_left > 0) {
    byte_sent = write(client_sock, buf, byte_left);
    byte_left -= byte_sent;
    buf += byte_sent;
  }

  // perform iteration 
  gettimeofday(&start, NULL);

  // Launch Kernel 
  void *kernelArgs[] = { &shared_buf, &N, &num_iter };
  cudaLaunchCooperativeKernel((void*)generateMat, dimGrid, dimBlock, kernelArgs);
  cudaEventRecord(kernelEnd, 0);
  cudaEventSynchronize(kernelEnd);

  gettimeofday(&end, NULL);
  timersub(&end, &start, &gap);
  double time = (double)gap.tv_sec + (double)gap.tv_usec / 1000000;
  printf("%d\n", (int)(time * 1000));

  // cleanup 
  close(client_sock);
  unlink(argv[1]);
  cudaEventDestroy(kernelEnd);
  cudaFree(shared_buf);

  return 0;
}


