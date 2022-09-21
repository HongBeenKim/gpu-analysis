#include <stdio.h>

extern "C"
{
__global__ void writeKernel(void *mapped_ptr, int size, unsigned char val) {
  for (int i = 0; i < size; i++) {
    *((unsigned int *)mapped_ptr) += 1;
  }
}

__global__ void readKernel(void *mapped_ptr, int size) {
  printf("value: %u\n", *((unsigned int *)mapped_ptr));

  /*
  for (int i = 0; i < 1024; i++) {
    printf(
      "value at %p+[%d]: 0x%x\n", 
      mapped_ptr, i, *((unsigned char *)mapped_ptr + i)
    );
  }
  */
}

}

