#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "touch_kernel.cuh"

int main() {
  cuInit(0);
  CUdevice dev;
  CUcontext ctx;
  cuDeviceGet(&dev, 0);
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);

  CUresult res;
  int *a = (int *)malloc(sizeof(int) * 1024);

  res = cuMemHostRegister((void *)a, 1024, CU_MEMHOSTREGISTER_DEVICEMAP);
  if (res) {
    printf("cuMemHostRegister Error: %d\n", res);
    exit(1);
  }

  int size = 1024;
  void *args[] = { &a, &size };
  cudaLaunchKernel((void*)touch, 1, 1, args, 0, NULL);
  cudaDeviceSynchronize();

  cuMemHostUnregister((void *)a);

  free(a);
  return 0;
}

