all : BlackScholes BlackScholesContention

BlackScholes : BlackScholes.cu BlackScholes_kernel.cuh
	nvcc -arch sm_80 BlackScholes.cu -o ./bin/BlackScholes

BlackScholesContention : BlackScholesContention.cu BlackScholes_kernel.cuh
	nvcc -arch sm_80 BlackScholesContention.cu -o ./bin/BlackScholesContention

