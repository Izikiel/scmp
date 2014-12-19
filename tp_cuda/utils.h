#include "stdio.h"
#include "cuda.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024 //la tesla tiene 1024 x bloque
#endif

#ifndef B2D_size
#define B2D_size 32
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((b) < (a) ? (a) : (b))
#endif

#ifndef COL
#define COL (threadIdx.x + blockDim.x * blockIdx.x)
#endif

#ifndef ROW
#define ROW (threadIdx.y + blockDim.y * blockIdx.y)
#endif

void print_matrix_msg(const char* msg, int* matrix, int N, int M);
void __cudaCheck(cudaError err, const char* file, const int line);

#define cudaCheck(err) __cudaCheck (err, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() cudaCheck(cudaDeviceSynchronize())