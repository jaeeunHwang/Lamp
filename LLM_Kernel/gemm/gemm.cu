/**********************************************************************
* FILENAME :       gemm_f32.cu
*
* DESCRIPTION :
*       Kernel side implementation of GEMM (General Matrix Multiplication)
*
* NOTES :
*
*
* AUTHOR :    Jaeeun Hwang, Eunbi Jeong, and Ikyoung Choi
*
* LAST MODIFIED : 2024/05/30
*
*********************************************************************/

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

// For debugging
#ifdef DEBUG_PRINTF
#define DEBUGPRINT(a) printf(a)
#else
#define DEBUGPRINT(a) (void)0
#endif

#define WARP_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define checkCudaError(error)                   \
        if(error != cudaSuccess){                               \
                printf("%s in %s at line %d\n", \
                                cudaGetErrorString(error),      \
                                __FILE__ ,__LINE__);                            \
                exit(EXIT_FAILURE);                                                     \
        }

#define TILE_WIDTH 32
__global__ void MatMul(float *A, float *B, float *C, int M, int K, int N) {
    // Shared memory for tile of matrix A and matrix B
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    // Thread indices
    int bx = blockIdx.x;    
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column indices of the element in matrix C to be computed
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Initialize the computed value to 0
    float Pvalue = 0;

    // Loop over tiles of matrix A and matrix B
    int ph;
    for (ph = 0; ph < K / TILE_WIDTH + 1; ph++) { // Loop over tiles along the shared dimension
        // Load data into shared memory for matrix A
        if (ph * TILE_WIDTH + tx < K) {
            Ads[ty][tx] = A[row * K + ph * TILE_WIDTH + tx];
        } else {
            Ads[ty][tx] = 0; // Zero-padding for out-of-bound indices
        }

        // Load data into shared memory for matrix B
        if (ph * TILE_WIDTH + ty < K) {
            Bds[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];
        } else {
            Bds[ty][tx] = 0; // Zero-padding for out-of-bound indices
        }

        // Synchronize threads to ensure all data is loaded into shared memory
        __syncthreads();

        // Compute the dot product of the tiles
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Ads[ty][k] * Bds[k][tx];
        }
        
        // Synchronize threads to ensure all threads have finished computing the dot product
        __syncthreads();
    }

    // Store the computed value in matrix C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = Pvalue;
    }
}

void readData(const char *file_name, float *values, int num) {
    FILE *fp = fopen(file_name, "rb");

    int cnt = 0;
    size_t len = 0;

    char delim[1];
    delim[0] = '\n';
    char *token;
    char *line = NULL;

    if (fp == NULL) {
        DEBUGPRINT(("FILE NOT FOUND ERROR\n"));
        return;
    } 
    
    else {
        DEBUGPRINT((" FILE FOUND\n"));
        
        while ((getline(&line, &len, fp)) != -1) {
            if ( cnt == num ) break;
            token = strtok(line, delim);
			    	values[cnt] = atof(token);
				    cnt++;
        }
        
        //printf("END OF WHILE : %d\n",cnt);
        
        for (; cnt < num; cnt++) {
	          values[cnt] = 0.0;
        }
     
        //printf("Final Token Count: %d\n", cnt);
    }
    
    fclose(fp);
}

void check_results(float *output, float *output_ref, int num) {
    int err = 0;
    for (int i = 0; i < num; i++) {
        if (fabs(output[i] - output_ref[i]) > 1e-5) {
            printf("Mismatch at index %d: GPU value = %f, Reference value = %f\n", i, output[i], output_ref[i]);
            err++;
        }
    }
    if (err == 0) {
        printf("All values are correct\n");
    } else {
        printf("Detected %d errors\n", err);
    }
}

namespace wt{
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
          
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
                             
  // JE : GMEM -> SMEM within A's BM * BK (column major)
  // 11008 % 8 = 1376, 4096 % 8 = 512
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    
    //const float8 tmp = reinterpret_cast<const float8 *>(&D[0])[0];
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    
 }

// JE : GMEM -> SMEM within B's BM * BK (row major)
  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
    
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) { //each SM
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    // JE : SMEM -> REG within A's WM * WK
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
    // JE : SMEM -> REG within B's WM * WK
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) { //warp 내 row
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) { //warp 내 col
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) { //resIdx 결과 index
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
    
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                (regM[wSubRowIdx * TM + resIdxM] * regN[wSubColIdx * TN + resIdxN]);

    
          }
        }
      }
    }
  }
}

}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    gemm(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  
  const uint cRow = blockIdx.y; // M / BM
  const uint cCol = blockIdx.x; // N / BN -> C's Block Tile

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARP_SIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN); // ThreadBlock이 몇 개의 warp로 나눠지는지
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER); //행렬 크기
  constexpr uint WSUBM = WM / WMITER; // 64/2=32 -> height
  constexpr uint WSUBN = WN / WNITER; // 32/2=16 -> width

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARP_SIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / (BK / 4); //스레드가 처리할 A의 행 결정
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK; //건너뛰어야 하는 행의 수
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }
     
    // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = (alpha * threadResults[i + 0]) + (beta * tmp.x);
	        tmp.y = (alpha * threadResults[i + 1]) + (beta * tmp.y);
	        tmp.z = (alpha * threadResults[i + 2]) + (beta * tmp.z);
	        tmp.w = (alpha * threadResults[i + 3]) + (beta * tmp.w);
                 
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
                         
        }
      }
    }
  }
  
}

void runGemm(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  cudaError_t err;
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 16;
  const uint WN = 64;
  const uint WM = 64;
  const uint WNITER = 4;
  const uint TN = 4;
  const uint TM = 8;
  dim3 blockDim(NUM_THREADS);

  constexpr uint NUM_WARPS = NUM_THREADS / WARP_SIZE;
  
  // warptile in threadblocktile
  static_assert((BN % WN == 0) and (BM % WM == 0));
  static_assert((BN / WN) * (BM / WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((WM * WN) % (WARP_SIZE * TM * TN * WNITER) == 0);
  constexpr uint WMITER = (WM * WN) / (32 * TM * TN * WNITER);
  // warpsubtile in warptile
  static_assert((WM % WMITER == 0) and (WN % WNITER == 0));

  static_assert((NUM_THREADS * 4) % BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((NUM_THREADS * 4) % BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(BN % (16 * TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(BM % (16 * TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((BM * BK) % (4 * NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");
                
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  gemm<BM, BN, BK, WM, WN, WNITER, TM,TN, NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

int main() {

  cudaError_t err;
  
  int M = 4096, N = 4096,  K = 4096;
  float alpha = 1.0, beta = 1.0; // GEMM input parameters, C=α*AB+β*C

  float *input_tensor = (float *)malloc(sizeof(float) *(M*K));
  float *weight_tensor = (float *)malloc(sizeof(float) *(K*N));
  float *output_tensor = (float *)malloc(sizeof(float) *(M*N));
  float *output_tensor2 = (float *)malloc(sizeof(float) *(M*N));
  float *output_tensor_ref = (float *)malloc(sizeof(float) *(M*N));
  
  readData("data/input.txt", input_tensor, M*K);
  readData("data/weight.txt", weight_tensor, K*N);
  readData("data/output.txt", output_tensor_ref, M*N);
  
  float *input_tensor_GPU;
  float *weight_tensor_GPU;
  float *output_tensor_GPU;
  float *output_tensor2_GPU;
  
  cudaMalloc((void **)&input_tensor_GPU, sizeof(float) *(M*K)); // 4096 * 11008
  cudaMalloc((void **)&weight_tensor_GPU, sizeof(float) *(K*N)); // 11008 * 4096
  cudaMalloc((void **)&output_tensor_GPU, sizeof(float) *(M*N)); // 4096 * 4096
  cudaMalloc((void **)&output_tensor2_GPU, sizeof(float) *(M*N)); // 4096 * 4096
  
  cudaMemcpy(input_tensor_GPU, input_tensor, sizeof(float) *(M*K),
                       cudaMemcpyHostToDevice);
  cudaMemcpy(weight_tensor_GPU, weight_tensor, sizeof(float) *(K*N),
                       cudaMemcpyHostToDevice);
    
  runGemm(M, N, K, alpha, input_tensor_GPU, weight_tensor_GPU, beta, output_tensor_GPU);
  err = cudaDeviceSynchronize();
  checkCudaError(err);

  cudaMemcpy(output_tensor, output_tensor_GPU, sizeof(float) *(M*N), cudaMemcpyDeviceToHost);
  
  const int THREAD_NUM_matmul = 32;
  const dim3 blockSize_matmul(THREAD_NUM_matmul, THREAD_NUM_matmul, 1);
  const dim3 gridSize_matmul((N+THREAD_NUM_matmul-1)/THREAD_NUM_matmul, (M+THREAD_NUM_matmul-1)/THREAD_NUM_matmul, 1);

  free(input_tensor);
  free(weight_tensor);
  free(output_tensor);
  free(output_tensor2);
  free(output_tensor_ref);
  
  cudaFree(input_tensor_GPU);
  cudaFree(weight_tensor_GPU);
  cudaFree(output_tensor_GPU);
  cudaFree(output_tensor2_GPU);

  return 0;
  
}
