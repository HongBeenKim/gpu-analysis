#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>

typedef CUresult (*orig_cuMemAddressReserve_f_type)( 
  CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags
);

CUresult cuMemAddressReserve(
  CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags
)
{
  orig_cuMemAddressReserve_f_type orig_cuMemAddressReserve;
  orig_cuMemAddressReserve = (orig_cuMemAddressReserve_f_type)dlsym(RTLD_NEXT, "cuMemAddressReserve");
  printf("cuMemAddressReserve called!\n");

  return orig_cuMemAddressReserve(ptr, size, alignment, addr, flags);
}

