#include "utils.h"

__global__ void reverse_array(int* arr, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N / 2) {
        int temp = arr[tid];
        arr[tid] = arr[N - 1 - tid];
        arr[N - 1 - tid] = temp;
    }
}

int main(int argc, char const* argv[])
{
    int N = 10;
    int* arr = (int*) malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        arr[i] = i;
    }

    int* dev_arr;
    cudaCheck(
        cudaMalloc((void**)&dev_arr, N * sizeof(int))
    );

    cudaCheck(
        cudaMemcpy(dev_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice)
    );

    dim3 grid_dim = dim3((N / BLOCK_SIZE + N % BLOCK_SIZE ? 1 : 0), 1, 1);
    dim3 block_dim = dim3(MIN(N / 2, BLOCK_SIZE), 1, 1);
    reverse_array <<< grid_dim, block_dim>>>(dev_arr, N);
    cudaCheck(
        cudaMemcpy(arr, dev_arr, N * sizeof(int), cudaMemcpyDeviceToHost)
    );

    printf("Array reverseado\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", arr[i] );
    }
    printf("\n");

    cudaCheck(
        cudaFree(dev_arr)
    );

    return 0;
}

