CC=nvcc

softmax: softmax.cu
	$(CC) -o softmax softmax.cu --cudart shared 
clean:
	rm -rf softmax
