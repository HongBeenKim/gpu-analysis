#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>


int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <port_file_name> <mat_size>\n", argv[0]);
    exit(1);
  }

  int N;
  int client_sock;
  struct sockaddr_un serveraddr;

  int *h_mat, *d_mat;
  
  char buf[512];
  bzero(buf, 512);

  sscanf(argv[2], "%d", &N);

  h_mat = (int *)malloc(sizeof(int) * N * N);
  if (cudaMalloc((void**)&d_mat, sizeof(int) * N * N)) {
    printf("cudaMalloc() failed\n");
    exit(1);
  }

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

  // receive matrix 
  read(client_sock, h_mat, sizeof(int) * N * N);

  cudaMemcpy(d_mat, h_mat, sizeof(int) * N * N, cudaMemcpyHostToDevice);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%2d ", h_mat[i * N + j]);
    }
    printf("\n");
  }

  close(client_sock);
  cudaFree(d_mat);
  free(h_mat);

  return 0;
}

