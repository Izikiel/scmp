#include "utils.h"

void __cudaCheck(cudaError err, const char* file, const int line)
{
    if ( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

void print_matrix_msg(const char* msg, int* matrix, int N, int M)
{
    printf("%s\n", msg);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%d ", matrix[j + i * M]);
        }
        printf("\n");
    }
    for (int i = 0; i < M; ++i) {
        printf("--");
    }
    printf("\n\n");
}