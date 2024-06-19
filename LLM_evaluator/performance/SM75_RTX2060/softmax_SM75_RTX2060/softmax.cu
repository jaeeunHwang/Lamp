#include <stdio.h>
#include <float.h>

#define WARP_SIZE 32

// For debugging
#ifdef DEBUG_PRINTF
#define DEBUGPINT(a) printf(a)
#else
#define DEBUGPRINT(a) (void)0
#endif

#define checkCudaError(error)                   \
        if(error != cudaSuccess){                               \
                printf("%s in %s at line %d\n", \
                                cudaGetErrorString(error),      \
                                __FILE__ ,__LINE__);                            \
                exit(EXIT_FAILURE);                                                     \
        }

// Utility function for softmax
// template<int WARPS_PER_BLOCK>
inline __device__ float block_sum(float *red_smem, float sum, int warps_per_block)
{
    // Decompose the thread index into warp/lane.
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_idx = threadIdx.x % WARP_SIZE;

    // Compute the sum per warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask);  
    }

    // Warp leaders store the data to shared memory.
    if (lane_idx == 0) {
        red_smem[warp_idx] = sum;
    }

    // Make sure the data is in shared memory. 
    __syncthreads();

    // The warps compute the final sums.
    if (lane_idx < warps_per_block) {
        sum = red_smem[lane_idx];
    }

    // Parallel reduction inside the warp.
#pragma unroll
    for (int mask = warps_per_block / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask);
    }

    // Broadcast to other threads.
    return __shfl_sync(0xFFFFFFFF, sum, 0);
}

__global__ void softmaxKernel(float *logits) 
{   
    int thread_idx = threadIdx.x;
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_idx = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int num_threads = blockDim.x;   // EB: equal to number of tokens

    DEBUGPRINT(("***** EB: FOR DEBUG *****\n"));
    DEBUGPRINT(("gridDim.x = %d, gridDim.y = %d, blockDim.x = %d, blockDim.y = %d, blockIdx.x = %d, blockIdx.y = %d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y));
    DEBUGPRINT(("thread_idx = %d, warp_idx = %d, lane_idx = %d\t", thread_idx, warp_idx, lane_idx));
    DEBUGPRINT(("num_warps = %d \n", num_warps));

    // Shared memory for reduction.
    __shared__ float red_smem[2 * WARP_SIZE];

    float qk_max = -FLT_MAX;
    float max_logit = logits[thread_idx + blockIdx.x * gridDim.x * gridDim.y];
    if (blockIdx.x == 0)
        DEBUGPRINT(("block #0: Thread %d - logits[%d] = %f\n", thread_idx, thread_idx + blockIdx.x * gridDim.x * gridDim.y, logits[thread_idx + blockIdx.x * gridDim.x * gridDim.y]));

    // Get the max qk value across the threads in the same warp 
    // (not across the thread block yet)
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(max_logit, __shfl_xor_sync(0xFFFFFFFF, qk_max, mask));
      	if (blockIdx.x == 0)
	    DEBUGPRINT(("block #0: Head 0 , Thread %d's qk_max = %f\n", thread_idx, qk_max));
    }
    if (blockIdx.x == 0)
        DEBUGPRINT(("block #0: first qk_max (not finished yet) = %f\n", qk_max));
    if (lane_idx == 0) {
        red_smem[warp_idx] = qk_max;
    }
    __syncthreads();

    // Get the max qk value for the sequence.
    // Synchronize the max qk value across different warps (butterfly reduction).
    qk_max = lane_idx < num_warps ? red_smem[lane_idx] : -FLT_MAX;
#pragma unroll    
    for (int mask = num_warps / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(0xFFFFFFFF, qk_max, mask));
    }
    qk_max = __shfl_sync(0xFFFFFFFF, qk_max, 0);
    if (blockIdx.x == 0)
        DEBUGPRINT(("block #0: qk_max 2 = %f\n", qk_max));
    
    // Calculate the exponential of (logits[i]-qk_max)
    float exp_sum = 0.f;
    int token_idx = thread_idx + blockIdx.x * blockDim.x * blockDim.y;

    for (int i = token_idx; i < token_idx + num_threads; i += blockDim.x) {
        float val = __expf(logits[i] - qk_max);
        if (blockIdx.x == 0)
            DEBUGPRINT(("block #0: exp(logits[%d]-qk_max) = %f\n", i, logits[i]));
        logits[i] = val;
        exp_sum += val;
    } 
    __syncthreads();

    if (blockIdx.x == 0)
        DEBUGPRINT(("block #0: logits[%d]-qk_max = %f, exp_sum = %f\n", token_idx, logits[token_idx], exp_sum));

    exp_sum = block_sum(&red_smem[num_warps], exp_sum, num_warps);

    if (blockIdx.x == 0)
        DEBUGPRINT(("EB: After block_sum, exp_sum = %f\n", exp_sum));

    // Normalize the logits.
    const float inv_sum = __fdividef(1.f, exp_sum + 1.e-6f);
    for (int i = token_idx ; i < token_idx + num_threads; i += blockDim.x) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    if (blockIdx.x == 0)
        DEBUGPRINT(("EB: logits = %f\n", logits[thread_idx]));
}

void readData(char *file_name, float *logits, int num_tokens) {
    FILE *fp = fopen(file_name, "rb");

    int cnt = 0;
    size_t len = 0;

    char delim[2] = {' ', 0};
    char *token;
    char *line = "\n";

    if (fp == NULL) {
        DEBUGPRINT(("[%s] FILE NOT FOUND ERROR\n", file_name));
        return;
    } else {
        DEBUGPRINT(("[%s] FILE FOUND\n", file_name));
        while ((getline(&line, &len, fp)) != -1) {
            token = strtok(line, delim);
            while (token != NULL) {
	    	logits[cnt] = (float)atof(token);
	    	token = strtok(NULL, " ");
		cnt++;
	    }
        }
        DEBUGPRINT(("Final Token Count: %dn", cnt));
    }
    fclose(fp);
}

void printLogits(char *data_name, float *logits, int num_prompts, int num_heads, int max_num_tokens) {
   int i, j, k;
   int token_cnt = 0;
   printf("***** Print out %s *****\n", data_name);
   for (i = 0; i < num_prompts; i++) {
       printf("[Prompt %d]\n", i);
       for (j = 0; j < num_heads; j++) {
           printf("- Head %d: ", j);
	   for (k = 0; k < max_num_tokens; k++) 
	      printf("%f ", logits[token_cnt++]);
	   printf("\n");  
       }
       printf("\n");
   }
}

int main() {
   char *input_file = "data/token_32/softmax_input.txt";
   int num_prompts = 1;
   int num_heads = 32;
   int max_num_tokens = 32; // EB: I assumed this number of tokens are same for every prompt.
   int num_tokens = num_prompts * num_heads * max_num_tokens;

   cudaError_t err;
   dim3 dimGrid(num_heads, num_prompts);
   dim3 dimBlock(max_num_tokens);

   float *h_logits, *d_logits, *h_scores; 
   h_logits = (float*)malloc(sizeof(float) * num_tokens);
   if (!h_logits) {
       printf("[ERROR] h_logits is null!\n");
       exit(1);
   } 
   readData(input_file, h_logits, num_tokens);
   printLogits("input", h_logits, num_prompts, num_heads, max_num_tokens);

   err = cudaMalloc((void**) &d_logits, sizeof(float) * num_tokens);
   checkCudaError(err);
   err = cudaMemcpy(d_logits, h_logits, sizeof(float) * num_tokens, cudaMemcpyHostToDevice);
   
   softmaxKernel<<<dimGrid, dimBlock>>>(d_logits);
   err = cudaDeviceSynchronize();
   checkCudaError(err);

   h_scores = (float*)malloc(sizeof(float) * num_tokens);
   if (!h_scores) {
   	printf("[ERROR] h_scores is null!\n");
   }
   err = cudaMemcpy(h_scores, d_logits, sizeof(float) * num_tokens, cudaMemcpyDeviceToHost);
   
   printLogits("output", h_scores, num_prompts, num_heads, max_num_tokens);

   free(h_logits);
   err = cudaFree(d_logits);
   checkCudaError(err);

   return 0;
}
