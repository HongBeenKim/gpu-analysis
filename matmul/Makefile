all : single double persistent

single : mm_single_stream.cu
	nvcc -arch sm_80 mm_single_stream.cu -o ./bin/single

double : mm_double_stream.cu
	nvcc -arch sm_80 mm_double_stream.cu -o ./bin/double

persistent : mm_persistent.cu
	nvcc -arch sm_80 mm_persistent.cu -o ./bin/persistent
