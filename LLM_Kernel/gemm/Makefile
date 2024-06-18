CC=nvcc

gemm: gemm.cu
	$(CC) -x cu -O3 -std=c++11 -arch=sm_80 -o gemm gemm.cu --cudart shared 
clean:
	rm -rf gemm
