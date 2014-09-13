#include "utils.h"

__global__ void max_arr(int* arr, int N, int* max)
{
    __shared__ int iter = 1;

    for (int i = N; i > 0; i >>= iter, iter++) {
        __syncthreads();

    }

}

int main(int argc, char const* argv[])
{
    int N = 10;
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int max;
    int* d_arr;
    int* d_max;
    cudaCheck(
        cudaMalloc((void**)&d_arr, N * sizeof(int))
    );
    cudaCheck(
        cudaMalloc((void**)&d_max, sizeof(int))
    );

    cudaCheck(
        cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice)
    );

    max_arr <<< 1, 10>>>(d_arr, N, d_max);

    cudaCheck(
        cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost)
    );
    cudaCheck(
        cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost)
    );

    printf("Maximo del array %d\n", max);

    cudaCheck(
        cudaFree(d_arr)
    );
    cudaCheck(
        cudaFree(d_max)
    );

    return 0;
}