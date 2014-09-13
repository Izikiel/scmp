#include "utils.h"

void __cudaCheck(cudaError err, const char* file, const int line)
{
    if ( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}