#include <cstdio>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include "kernel.cuh"

int main() {
  cudaError_t err;

  int size = 268435456, byteSent;
  int *devPtrs[3];
  cudaIpcMemHandle_t memHandle;

  int sockfd = socket(AF_UNIX, SOCK_DGRAM, 0);

  unlink("server");

  struct sockaddr_un serveraddr, clientaddr;
  serveraddr.sun_family = AF_UNIX;
  strcpy(serveraddr.sun_path, "server");
  clientaddr.sun_family = AF_UNIX;
  strcpy(clientaddr.sun_path, "client");

  if (bind(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) < 0) {
    perror("bind() error: ");
    exit(1);
  }

  for (int i = 0; i < 3; i++) {
    if ((err = cudaMalloc(&devPtrs[i], sizeof(int) * size))) {
      printf("cudaMalloc() failed (%d)\n", err);
      return 1;
    }

    if ((err = cudaIpcGetMemHandle(&memHandle, devPtrs[i]))) {
      printf("cudaIpcGetMemHandle() failed (%d)\n", err);
      return 1;
    }

    byteSent = sendto(
      sockfd, (void *)&memHandle, 
      sizeof(memHandle), 0, 
      (struct sockaddr *)&clientaddr, sizeof(clientaddr)
    );
  }

  fill<<<524288, 512>>>(devPtrs[0], size, 1);
  if (cudaPeekAtLastError()) {
    printf("Kernel launch failed\n");
  }
  fill<<<524288, 512>>>(devPtrs[1], size, 2);
  if (cudaPeekAtLastError()) {
    printf("Kernel launch failed\n");
  }
  cudaDeviceSynchronize();

  char sig;
  socklen_t clientAddrLen;
  byteSent = recvfrom(
    sockfd, (void*)&sig, 1, 0,
    (struct sockaddr *)&clientaddr, &clientAddrLen
  );

  struct timeval start, end, interval;
  gettimeofday(&start, NULL);

  /* ------------- Start ------------- */
  add<<<262144, 512>>>(devPtrs[0], devPtrs[1], devPtrs[2], size / 2);
  if (cudaPeekAtLastError()) {
    printf("Kernel launch failed\n");
  }
  cudaDeviceSynchronize();

  byteSent = recvfrom(
    sockfd, (void*)&sig, 1, 0,
    (struct sockaddr *)&clientaddr, &clientAddrLen
  );
  /* -------------- End -------------- */

  gettimeofday(&end, NULL);
  timersub(&end, &start, &interval);
  printf("%ld.%06ld\n", (long int)interval.tv_sec, (long int)interval.tv_usec);

  check<<<524288, 512>>>(devPtrs[2], size, 3);
  if (cudaPeekAtLastError()) {
    printf("Kernel launch failed\n");
  }

  for (int i = 0; i < 3; i++) {
    cudaFree(devPtrs[i]);
  }
  close(sockfd);
  return 0;
}

