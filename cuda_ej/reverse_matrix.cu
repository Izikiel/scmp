#include "stdio.h"
#include "cuda.h"

__global__ void reverse_matrixY(int *matrix, int N)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx < N / 2) {
        int temp = matrix[tx + ty * N];
        matrix[tx + ty * N] = matrix[N - 1 - tx + ty * N];
        matrix[N - 1 - tx + ty * N] = temp;
    }
}

__global__ void reverse_matrixX(int *matrix, int N)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (ty < N / 2) {
        int temp = matrix[ty + tx * N];
        matrix[ty + tx * N] = matrix[N - 1 - ty + tx * N];
        matrix[N - 1 - ty + tx * N] = temp;
    }
}

void print_matrix_msg(const char *msg, int *matrix, int N)
{
    printf("%s\n", msg);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_matrix[j + i * N]);
        }
        printf("\n");
    }
    for (int i = 0; i < N; ++i) {
        printf("--");
    }
    printf("\n\n");

}

int main(int argc, char const *argv[])
{
    int N = 10;

    int *h_matrix = malloc(N * N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_matrix[j + i * N] = j;
        }
    }

    print_matrix_msg("Matriz sin reversear\n", h_matrix, N);

    int *d_matrix;

    cudaMalloc((void **)&d_matrix, N * N * sizeof(int));
    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice);

    reverse_matrixY <<< dimGrid, dimBlock >>>(d_matrix, N);

    cudaMemcpy(h_matrix, d_matrix, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    print_matrix_msg("Reverseo en Y", h_matrix, N);


    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_matrix[i + j * N] = j;
        }
    }
    print_matrix_msg("Matriz sin reversear\n", h_matrix, N);

    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice);

    reverse_matrixX <<< dimGrid, dimBlock >>>(d_matrix, N);

    cudaMemcpy(h_matrix, d_matrix, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    print_matrix_msg("Reverseo en X", h_matrix, N);


    cudaFree(d_matrix);
    free(h_matrix);
    return 0;
}