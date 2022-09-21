#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string.h>
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

  int sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0);

  struct sockaddr_un un;
  un.sun_family = AF_UNIX;
  unlink("sockFile");
  strcpy(un.sun_path, "sockFile");

  if (bind(sock_fd, (struct sockaddr *)&un, sizeof(un)) < 0) {
    printf("bind() failed\n");
    return 1;
  }

  char buf[512];
  struct iovec e = { buf, 512 };
  char cmsg[CMSG_SPACE(sizeof(int))];
  struct msghdr m = { NULL, 0, &e, 1, cmsg, sizeof(cmsg), 0 };

  recvmsg(sock_fd, &m, 0);

  struct cmsghdr *c = CMSG_FIRSTHDR(&m);
  int cuMemFd = *(int*)CMSG_DATA(c); // receive file descriptor

  CUmemGenericAllocationHandle cuMemHandle;
  res = cuMemImportFromShareableHandle(
    &cuMemHandle, (void*)(uintptr_t)cuMemFd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
  );
  if (res != 0) {
    printf("cuMemImportFromShareableHandle() failed (%d)\n", res);
    return 1;
  }

  int size = 2097152;
  CUdeviceptr devPtr;
  cuMemAddressReserve(&devPtr, size, 0, 0, 0);
  res = cuMemMap(devPtr, size, 0, cuMemHandle, 0);
  if (res != 0) {
    printf("cuMemMap() failed (%d)\n", res);
    return 1;
  }

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = dev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess(devPtr, size, &accessDesc, 1);

  dim3 d = { 1, 1, 1 };
  // Lauhcn kernel here
  unsigned char val = 0xaa;
  int cnt = 10000000;
  void *args[] = { &devPtr, &cnt, &val };
  cudaError_t err = cudaLaunchKernel((void*)writeKernel, d, d, args, 0, NULL);
  printf("error: %d\n", err);
  cudaDeviceSynchronize();

  void *args2[] = { &devPtr, &size };
  err = cudaLaunchKernel((void*)readKernel, d, d, args2, 0, NULL);
  cudaDeviceSynchronize();

  return 0;
}

