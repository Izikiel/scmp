#include "utils.h"

__global__ void reverse_matrixY(int* matrix, int N)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < N / 2) {
        int temp = matrix[col + row * N];
        matrix[col + row * N] = matrix[N - 1 - col + row * N];
        matrix[N - 1 - col + row * N] = temp;
    }
}

__global__ void reverse_matrixX(int* matrix, int N)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < N / 2) {
        int temp = matrix[row + col * N];
        matrix[row + col * N] = matrix[N - 1 - row + col * N];
        matrix[N - 1 - row + col * N] = temp;
    }
}

void print_matrix_msg(const char* msg, int* matrix, int N)
{
    printf("%s\n", msg);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", matrix[j + i * N]);
        }
        printf("\n");
    }
    for (int i = 0; i < N; ++i) {
        printf("--");
    }
    printf("\n\n");

}

int main(int argc, char const* argv[])
{
    int N = 10;

    int* h_matrix = (int*) malloc(N * N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_matrix[j + i * N] = j;
        }
    }

    print_matrix_msg("Matriz sin reversear\n", h_matrix, N);

    int* d_matrix;
    unsigned int blocks = (N / BLOCK_SIZE + (N % BLOCK_SIZE ? 1 : 0));
    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlockY(MIN(N / 2, BLOCK_SIZE), MIN(N, BLOCK_SIZE), 1);

    cudaCheck(
        cudaMalloc((void**)&d_matrix, N * N * sizeof(int))
    );

    cudaCheck(
        cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice)
    );

    reverse_matrixY <<< dimGrid, dimBlockY >>>(d_matrix, N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_matrix[i * N + j] = 0;
        }
    }

    cudaCheck(
        cudaMemcpy(h_matrix, d_matrix, N * N * sizeof(int), cudaMemcpyDeviceToHost)
    );

    print_matrix_msg("Reverseo en Y", h_matrix, N);


    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         h_matrix[i + j * N] = j;
    //     }
    // }
    // print_matrix_msg("Matriz sin reversear\n", h_matrix, N);

    // cudaCheck(
    //     cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice)
    // );
    // dim3 dimBlockX(MIN(N, BLOCK_SIZE), MIN(N/2, BLOCK_SIZE), 1);

    // reverse_matrixX <<< dimGrid, dimBlockX >>>(d_matrix, N);

    // cudaCheck(
    //     cudaMemcpy(h_matrix, d_matrix, N * N * sizeof(int), cudaMemcpyDeviceToHost)
    // );

    // print_matrix_msg("Reverseo en X", h_matrix, N);


    cudaFree(d_matrix);
    free(h_matrix);
    return 0;
}