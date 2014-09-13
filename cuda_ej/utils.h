#include "stdio.h"
#include "cuda.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

void __cudaCheck(cudaError err, const char* file, const int line);
#define cudaCheck(err) __cudaCheck (err, __FILE__, __LINE__)