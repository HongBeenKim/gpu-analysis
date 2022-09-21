#include <sys/socket.h>
#include <sys/un.h>
#include <cstdio>
#include <unistd.h>
#include "kernel.cuh"

int main() {
  cudaError_t err;
  int byteReceived;
  int sockfd = socket(AF_LOCAL, SOCK_DGRAM, 0);
  int size = 268435456;
  socklen_t serverAddrLen;
  
  unlink("client");
  struct sockaddr_un clientaddr, serveraddr;
  clientaddr.sun_family = AF_UNIX;
  strcpy(clientaddr.sun_path, "client");

  if (bind(sockfd, (struct sockaddr *)&clientaddr, sizeof(clientaddr)) < 0) {
    perror("bind() error: ");
    exit(1);
  }

  int *devPtrs[3];
  cudaIpcMemHandle_t memHandle;

  for (int i = 0; i < 3; i++) {
    byteReceived = recvfrom(
      sockfd, (void *)&memHandle, 
      sizeof(memHandle), 0,
      (struct sockaddr *)&serveraddr, 
      &serverAddrLen
    );

    err = cudaIpcOpenMemHandle(
      (void**)&devPtrs[i], memHandle, cudaIpcMemLazyEnablePeerAccess
    );
    if (err) {
      printf("cudaIpcOpenMemHandle() failed (%d)\n", err);
      exit(1);
    }
  }

  char sig = 's';
  byteReceived = sendto(
    sockfd, &sig, 1, 0, 
    (struct sockaddr *)&serveraddr,
    serverAddrLen
  );

  printf("launching half kernel 2\n");
  add<<<262144, 512>>>(
    devPtrs[0] + size / 2, 
    devPtrs[1] + size / 2,
    devPtrs[2] + size / 2, 
    size / 2
  );
  cudaDeviceSynchronize();
  printf("half kernel 2 terminated\n");

  sig = 'f';
  byteReceived = sendto(
    sockfd, &sig, 1, 0, 
    (struct sockaddr *)&serveraddr,
    serverAddrLen
  );

  for (int i = 0; i < 3; i++) {
    cudaIpcCloseMemHandle(devPtrs[i]);
  }
  close(sockfd);

  return 0;
}

