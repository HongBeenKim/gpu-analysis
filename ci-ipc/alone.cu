#include <cstdio>
#include <unistd.h>
#include <sys/time.h>
#include "kernel.cuh"

int main() {
  cudaError_t err;

  int size = 268435456;
  int *devPtrs[3];

  for (int i = 0; i < 3; i++) {
    if ((err = cudaMalloc(&devPtrs[i], sizeof(int) * size))) {
      printf("cudaMalloc() failed (%d)\n", err);
      return 1;
    }
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

  struct timeval start, end, interval;
  gettimeofday(&start, NULL);

  /* ------------- Start ------------- */
  add<<<524288, 512>>>(devPtrs[0], devPtrs[1], devPtrs[2], size);
  if (cudaPeekAtLastError()) {
    printf("Kernel launch failed\n");
  }
  cudaDeviceSynchronize();
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

  return 0;
}

