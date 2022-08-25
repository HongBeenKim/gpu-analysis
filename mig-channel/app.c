#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <sys/ioctl.h>
#include <asm/types.h>
#include "drv/mig_channel.h"


int pin_buffer(int fd, CUdeviceptr addr, size_t size) {
  MIG_CHANNEL_IOC_PIN_BUFFER_PARAMS params;
  params.addr = (unsigned long)addr;
  params.size = size;

  return ioctl(fd, MIG_CHANNEL_IOC_PIN_BUFFER, &params);
}

int unpin_buffer(int fd) {
  return ioctl(fd, MIG_CHANNEL_IOC_UNPIN_BUFFER, NULL);
}

int read_both(int fd) {
  return ioctl(fd, MIG_CHANNEL_IOC_READ_BOTH, NULL);
}

int write_both(int fd) {
  MIG_CHANNEL_IOC_PIN_WRITE_BOTH_PARAMS params;
  params.value = 0xab;
  return ioctl(fd, MIG_CHANNEL_IOC_WRITE_BOTH, &params);
}

int main() {
  int channelDrvFd = open("/dev/migchannel", O_RDWR | O_CLOEXEC);
  if (channelDrvFd < 0) {
    printf("Failed to open /dev/migchannel inode\n");
    exit(1);
  }
  int allocSize = 65536;

  CUresult res;
  res = cuInit(0);
  if (res) {
    printf("cuInit failed\n");
    exit(1);
  }

  CUcontext ctx;
  res = cuCtxCreate(&ctx, 0, 0);

  CUdeviceptr devPtr;
  res = cuMemAlloc(&devPtr, allocSize);

  int r = pin_buffer(channelDrvFd, devPtr, allocSize);
  r = write_both(channelDrvFd);
  r = read_both(channelDrvFd);
  r = unpin_buffer(channelDrvFd);

  cuMemFree(devPtr);
  close(channelDrvFd);
  return 0;
}

