all : cache mps

cache : mm_normal.cu
	nvcc mm_normal.cu -o cache

mps : mm_mps.cu
	nvcc mm_mps.cu -o mps

