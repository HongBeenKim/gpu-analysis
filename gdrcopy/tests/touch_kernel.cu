#include <stdio.h>

__global__ void touch(void *mapped_ptr, int size) {
  for (int i = 0; i < 4096; i++) {
    *((unsigned char *)mapped_ptr + i) = 0xcc;
  }

  for (int i = 0; i < 4096; i++) {
    printf(
      "value at %p+[%d]: %u\n", 
      mapped_ptr, i, *((unsigned char *)mapped_ptr + i)
    );
  }
}


