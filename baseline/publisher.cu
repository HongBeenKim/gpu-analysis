#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <curand_kernel.h>


__global__ void generateMat(int *mat, int N) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int gridLen = gridDim.x * blockDim.x;

  int seed = gridLen * row + col;
  curandState_t state;
  curand_init(seed, 0, 0, &state);

  float rand_int = curand_uniform(&state);
  rand_int = rand_int * 20;

  for (int i = row; i < N; i += gridLen) {
    for (int j = col; j < N; j += gridLen) {
      mat[i * N + j] = (int)rand_int;
    }
  }
}


int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <port_file_name> <mat_size>\n", argv[0]);
    exit(1);
  }

  int N;
  int num_subscribers = 1;
  int server_sock, client_sock;
  struct sockaddr_un serveraddr, clientaddr;
  socklen_t client_len = sizeof(clientaddr);

  dim3 dimGrid(4, 4);
  dim3 dimBlock(16, 16);

  char buf[512];
  int *h_mat, *d_mat;

  sscanf(argv[2], "%d", &N);

  cudaEvent_t kernelEnd;
  cudaEventCreate(&kernelEnd);

  h_mat = (int *)malloc(sizeof(int) * N * N);
  if (cudaMalloc((void**)&d_mat, sizeof(int) * N * N)) {
    printf("cudaMalloc() failed\n");
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

  // perform iteration 

  generateMat<<<dimGrid, dimBlock>>>(d_mat, N);
  cudaEventRecord(kernelEnd, 0);
  cudaEventSynchronize(kernelEnd);

  cudaMemcpy(h_mat, d_mat, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

  // send matrix 
  write(client_sock, h_mat, sizeof(int) * N * N);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%2d ", h_mat[i * N + j]);
    }
    printf("\n");
  }

  // cleanup 
  close(client_sock);
  unlink(argv[1]);
  cudaEventDestroy(kernelEnd);
  cudaFree(d_mat);
  free(h_mat);

  return 0;
}

