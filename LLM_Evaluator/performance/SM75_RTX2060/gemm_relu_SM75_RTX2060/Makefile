CC=nvcc

gemm_relu: gemm_relu.cu
	$(CC) -x cu -O3 -std=c++11 -arch=sm_80 -o gemm_relu gemm_relu.cu --cudart shared
clean:
	rm -rf gemm_relu
