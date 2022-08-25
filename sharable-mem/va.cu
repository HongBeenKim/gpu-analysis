#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <cuda.h>
#include <mpi.h>

__global__ void touch(void *mapped_ptr) {
  char *c = (char *)mapped_ptr;
  *c = 0x2;
  // printf("value: 0x%x\n", *c);
}

int main(int argc, char *argv[]) {
  CUresult res;
  int allocSize = 1024;

  res = cuInit(0);
  if (res) {
    printf("cuInit failed\n");
    exit(1);
  }

  CUcontext ctx;
  res = cuCtxCreate(&ctx, 0, 0);

  //CUdeviceptr ptr;
  //res = cuMemAlloc(&ptr, allocSize);
  void *ptr;
  res = cuMemHostAlloc(&ptr, allocSize, CU_MEMHOSTALLOC_DEVICEMAP);

  printf("ptr: 0x%llx\n", ptr);

  void *args[] = { &ptr };

  cudaLaunchKernel((void*)touch, 1, 1, args, 0, NULL);
  // res = cuLaunchKernel(touch, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);

  // res = cuMemFree(ptr);

/*
  CUmemGenericAllocationHandle handle;

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;

  size_t granularity = 0;
  cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  
  res = cuMemCreate(&handle, granularity * 32, &prop, 0);
  if (res) {
    printf("cuMemCreate failed (res: %d)\n", res);
    exit(1);
  }

  CUdeviceptr ptr;
  res = cuMemAddressReserve(&ptr, granularity * 32, 0, 0, 0);
  if (res) {
    printf("cuMemAddressReserve failed (res: %d)\n", res);
    exit(1);
  }

  res = cuMemMap(ptr, granularity * 32, 0, handle, 0);

  res = cuMemRelease(handle);
  if (res) {
    printf("cuMemRelease failed (res: %d)\n", res);
    exit(1);
  }
*/
  /*
  res = cuMemExportToShareableHandle(
    (void *)&shrHandle, handle, 
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0
  );
  if (res)
    printf("cuMemExportToShareableHandle fail: %d\n", res);
  */

  return 0;
}

