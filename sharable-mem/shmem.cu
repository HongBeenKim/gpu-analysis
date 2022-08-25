#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <cuda.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  /*
  if (argc < 3) {
    printf("Usage: %s <array_size> <direction>\n", argv[0]);
    exit(1);
  }
  */

  const char GIList[2][42] = {
    "MIG-6e5ecf1c-980b-53b4-b79e-df70177fd284",
    "MIG-3234bc3b-83f3-5e3a-940e-d1c72da74e00"
  };

  MPI_Init(&argc, &argv);

  int processCount;
  MPI_Comm_size(MPI_COMM_WORLD, &processCount);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //setenv("CUDA_VISIBLE_DEVICES", GIList[rank], 1);
  CUresult res;
  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);

  //CUcontext ctx;
  //cuCtxCreate(&ctx, 0, dev);

  CUmemGenericAllocationHandle handle;
  int shrHandle;

  if (rank) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    
    res = cuMemCreate(&handle, granularity, &prop, 0);
    if (res)
      printf("cuMemCreate fail: %d\n", res);

    res = cuMemExportToShareableHandle(
      (void *)&shrHandle, handle, 
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0
    );
    if (res)
      printf("cuMemExportToShareableHandle fail: %d\n", res);

    printf("Sending shHandle: %d\n", shrHandle);
    MPI_Send(&shrHandle, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    sleep(5);

  } else {
    MPI_Recv(&shrHandle, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("received shHandle: %d\n", shrHandle);

    res = cuMemImportFromShareableHandle(
      &handle, (void *)24,//(void *)(uintptr_t)shrHandle, 
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    );
    printf("ret: %d\n", res);
  }

  if (rank) {
    cuMemRelease(handle);
  }

  MPI_Finalize();
  return 0;
}

