#include <stdio.h>

__global__ void touch(void *mapped_ptr, int size) {
  for (int i = 0; i < size; i++) {
    *((unsigned char *)mapped_ptr + i) = 0xcc;
  }

  for (int i = 0; i < size; i++) {
    printf(
      "value at %p+[%d]: 0x%x\n",
      mapped_ptr, i, *((unsigned char *)mapped_ptr + i)
    );
  }
}

