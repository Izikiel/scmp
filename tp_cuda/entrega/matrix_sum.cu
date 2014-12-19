#include "utils.h"

void infoDevices();
void printDevProp(cudaDeviceProp devProp);


__global__ void matrix_sum(int* A, int* B, int* C, int numCol, int numRow)
{
    uint pos = COL + ROW * numCol;
    if (pos < numCol * numRow) {
        C[pos] = A[pos] + B[pos];
    }
}

void infoDevices()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    for (int i = 0; i < devCount; ++i) {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
}

void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i) {
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    }
    for (int i = 0; i < 3; ++i) {
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    }
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}


int main(int argc, char const* argv[])
{
    //MEDIR TIEMPOS :D
    if (argc < 4) {
        printf("Tamaño matrix, cantidad de bloques, threads x bloque\n");
        return 2;
    }

    // infoDevices();

    int N, M;
    sscanf(argv[1], "%d", &N);
    M = N;

    unsigned int blocks;
    sscanf(argv[2], "%d", &blocks);

    int threads;
    sscanf(argv[3], "%d", &threads);

    dim3 dimGrid(blocks, blocks , 1);
    dim3 dimBlock(threads, threads, 1);

    int size = N * M * sizeof(int);

    int* hA = (int*) calloc(N * M, sizeof(int));
    int* hB = (int*) calloc(N * M, sizeof(int));
    int* hC = (int*) calloc(N * M, sizeof(int));

    int* dA;
    int* dB;
    int* dC;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaCheck(
        cudaMalloc((void**)&dA, size)
    );
    cudaCheck(
        cudaMalloc((void**)&dB, size)
    );
    cudaCheck(
        cudaMalloc((void**)&dC, size)
    );

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            hA[j + i * M] = i;
            hB[j + i * M] = j;
        }
    }

    // print_matrix_msg("A", hA, N, M);
    // print_matrix_msg("B", hB, N, M);

    cudaCheck(
        cudaMemcpy(dA, hA, N * M * sizeof(int), cudaMemcpyHostToDevice)
    );
    cudaCheck(
        cudaMemcpy(dB, hB, N * M * sizeof(int), cudaMemcpyHostToDevice)
    );
    cudaCheck(
        cudaMemcpy(dC, hC, N * M * sizeof(int), cudaMemcpyHostToDevice)
    );


    cudaEventRecord(start, 0);

    matrix_sum <<< dimGrid, dimBlock>>>(dA, dB, dC, N, M);

    CUDA_CHECK_ERROR();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;

    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaCheck(
        cudaMemcpy(hC, dC, N * M * sizeof(int), cudaMemcpyDeviceToHost)
    );

    // print_matrix_msg("C = A + B", hC, N, M);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            if (hC[j + i * M] != (hA[j + i * M] + hB[j + i * M]) ) {
                return 1;
            }
        }
    }

    cudaEventRecord(start, 0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            hC[j + i * M] = hA[j + i * M] + hB[j + i * M];
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cpu_time;

    cudaEventElapsedTime(&cpu_time, start, stop);

    printf("%g %g\n", gpu_time, cpu_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(hA);
    free(hB);
    free(hC);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;


}