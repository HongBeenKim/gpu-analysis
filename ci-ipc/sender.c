#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/un.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "touch_kernel.cuh"

int main() {
  CUresult res;
  cuInit(0);

  CUdevice dev;
  cuDeviceGet(&dev, 0);

  CUcontext ctx;
  cuCtxCreate(&ctx, 0, dev);
  // cuDevicePrimaryCtxRetain(&ctx, dev);
  // cuCtxSetCurrent(ctx);

  CUmemGenericAllocationHandle cuMemHandle;
  CUmemAllocationProp prop = {
    .type = CU_MEM_ALLOCATION_TYPE_PINNED,
    .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    .location.type = CU_MEM_LOCATION_TYPE_DEVICE,
    .location.id = 0
  };

  size_t granularity = 0;
  cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

  res = cuMemCreate(&cuMemHandle, granularity, &prop, 0);
  if (res != 0) {
    printf("cuMemCreate() error\n");
    return 1;
  }

  int memHandleFd;
  res = cuMemExportToShareableHandle(
    (void*)&memHandleFd, cuMemHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0
  );
  if (res != 0) {
    printf("cuMemExportToShareableHandle() failed\n");
    return 1;
  }

  int sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0);

  struct sockaddr_un ad;
  ad.sun_family = AF_UNIX;
  strcpy(ad.sun_path, "sockFile");

  struct iovec e = { NULL, 0 };
  char cmsg[CMSG_SPACE(sizeof(int))];

  struct msghdr m = { (void*)&ad, sizeof(ad), &e, 1, cmsg, sizeof(cmsg), 0 };

  struct cmsghdr *c = CMSG_FIRSTHDR(&m);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(sizeof(int));
  *(int*)CMSG_DATA(c) = memHandleFd; // set file descriptor

  sendmsg(sock_fd, &m, 0);

  CUdeviceptr devPtr;
  cuMemAddressReserve(&devPtr, granularity, 0, 0, 0);
  res = cuMemMap(devPtr, granularity, 0, cuMemHandle, 0);
  if (res != 0) {
    printf("cuMemMap() failed\n");
    return 1;
  }
  
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = dev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess(devPtr, granularity, &accessDesc, 1);

  dim3 d = { 1, 1, 1 };
  void *ptr = (void*)devPtr;
  // Lauhcn kernel here 
  unsigned char val = 0xbb;
  int cnt = 10000000;
  void *args[] = { &devPtr, &cnt, &val };
  cudaError_t err = cudaLaunchKernel((void*)writeKernel, d, d, args, 0, NULL);
  printf("error: %d\n", err);
  cudaDeviceSynchronize();

  close(sock_fd);
  cuMemUnmap(devPtr, granularity);
  cuMemRelease(cuMemHandle);

  return 0;
}

