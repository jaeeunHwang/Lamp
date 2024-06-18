CC=nvcc

softmax: softmax.cu
	$(CC) -x cu -O3 -std=c++11 -arch=sm_80 -o softmax softmax.cu --cudart shared 
clean:
	rm -rf softmax
