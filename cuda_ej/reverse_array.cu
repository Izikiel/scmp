#include "stdio.h"
#include "cuda.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
__global__ void reverse_array(int *arr, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N / 2) {
        int temp = arr[tid];
        arr[tid] = arr[N - 1 - tid];
        arr[N - 1 - tid] = temp;
    }
}

int main(int argc, char const *argv[])
{
    int N = 10;
    int *arr = malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        arr[i] = i;
    }

    int *dev_arr;
    cudaMalloc((void **)&dev_arr, N * sizeof(int));
    cudaMemcpy(dev_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid_dim = dim3((N / BLOCK_SIZE + N % BLOCK_SIZE ? 1 : 0), 0);
    dim3 block_dim = dim3(MIN(N / 2, BLOCK_SIZE));
    reverse_array <<< grid_dim, block_dim>>>(dev_arr, N);
    cudaMemcpy(arr, dev_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Array reverseado\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", arr[i] );
    }
    printf("\n");

    cudaFree(dev_arr);

    return 0;
}

