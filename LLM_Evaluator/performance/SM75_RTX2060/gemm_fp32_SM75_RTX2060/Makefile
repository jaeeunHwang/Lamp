CC=nvcc

gemm: gemm_fp32.cu
	$(CC) -x cu -O3 -std=c++11 -arch=sm_80 -o gemm_fp32 gemm_fp32.cu --cudart shared 
clean:
	rm -rf gemm_fp32
