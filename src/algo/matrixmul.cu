#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <assert.h>
#include <fixedlonglong32x32.cuh>
#include <functional.h>
#include <kernels.cuh>


void maxmulFloat(float *A, float *B, float *C, int m, int n, int k) {

    // Allocate device memory:
    float *gpu_A;
    float *gpu_B;
    float *gpu_C;

    cudaMalloc((void **) &gpu_A, sizeof(float) * m * n);
    cudaMemcpy(gpu_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &gpu_B, sizeof(float) * n * k);
    cudaMemcpy(gpu_B, B, sizeof(float) * n * k, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &gpu_C, sizeof(float) * m * k);

    int BLOCK_SIZE = 16;
    // Blocks & grids:
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(grid_cols, grid_rows);

    // Call the kernel:
    vecmulFloat<<<grid, blocks>>>(gpu_A, gpu_B, gpu_C, m, n, k);

    // Get the result Matrix:
    cudaMemcpy(C, gpu_C, sizeof(float) * m * k, cudaMemcpyDeviceToHost);

    //Free device matrices
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
}

void maxmulFixedLongLong(long long *A, long long *B, long long *C, int m, int n, int k) {
    // cout << "Start maxmulFixedLongLong\n";

    // Allocate device memory:
    long long *gpu_A;
    long long *gpu_B;
    long long *gpu_C;

    // cout << "------------------\n";
    // cout << "A = {";
    //for (int i = 0; i < m * n; ++i) cout << A[i] << (i != m * n - 1 ? ", " : "}\n");
    cudaMallocManaged(&gpu_A, sizeof(long long) * m * n);
    // cout << "cudaMallocManaged(&gpu_A, sizeof(long long)*m*n)\n";

    cudaError_t errorA = cudaMemcpy(gpu_A, A, sizeof(long long) * m * n, cudaMemcpyHostToDevice);
    //cout << "error cudaError_t cudaMemcpy gpu_A " << cudaGetErrorString(errorA) << "\n";
    //cout << "gpu_A = {";
    //for (int i = 0; i < m * n; ++i) cout << *(gpu_A+i) << (i != m * n - 1 ? ", " : "}\n");

    cudaMallocManaged(&gpu_B, sizeof(long long) * n * k);
    // cout << "B = {";
    // for (int i = 0; i < n * k; ++i) cout << B[i] << (i != n * k - 1 ? ", " : "}\n");
    cudaError_t errorB = cudaMemcpy(gpu_B, B, sizeof(long long) * n * k, cudaMemcpyHostToDevice);
    //cout << "error cudaError_t cudaMemcpy gpu_B " << cudaGetErrorString(errorB) << "\n";
    //cout << "gpu_B = {";
    //for (int i = 0; i < n * k; ++i) cout << *(gpu_B+i) << (i != n * k - 1 ? ", " : "}\n");

    cudaMallocManaged(&gpu_C, sizeof(long long) * m * k);

    int BLOCK_SIZE = 16;
    // Blocks & grids:
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(grid_cols, grid_rows);

    // Call the kernel:
    vecmulFixedLongLong<<<grid, blocks>>>(gpu_A, gpu_B, gpu_C, m, n, k);


    //cout << "gpu_C = {";
    //for (int i = 0; i < m * k; ++i) cout << *(gpu_C+i) << (i != m * k - 1 ? ", " : "}\n");
    // Get the result Matrix:
    cudaMemcpy(C, gpu_C, sizeof(long long) * m * k, cudaMemcpyDeviceToHost);
    // cout << "C = {";
    // for (int i = 0; i < m * k; ++i) cout << C[i] << (i != m * k - 1 ? ", " : "}\n");

    // cout << "------------------\n";
    //Free device matrices
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    // cout << "End maxmulFixedLongLong\n";
}

void maxmulLong(long *A, long *B, long *C, long m, long n, long k) {

    // Allocate device memory:
    long *gpu_A;
    long *gpu_B;
    long *gpu_C;

    cudaMalloc((void **) &gpu_A, sizeof(long) * m * n);
    cudaMemcpy(gpu_A, A, sizeof(long) * m * n, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &gpu_B, sizeof(long) * n * k);
    cudaMemcpy(gpu_B, B, sizeof(long) * n * k, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &gpu_C, sizeof(long) * m * k);

    int BLOCK_SIZE = 16;
    // Blocks & grids:
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(grid_cols, grid_rows);

    // Call the kernel:
    vecmulLong<<<grid, blocks>>>(gpu_A, gpu_B, gpu_C, m, n, k);

    // Get the result Matrix:
    cudaMemcpy(C, gpu_C, sizeof(long) * m * k, cudaMemcpyDeviceToHost);

    //Free device matrices
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
}

void maxmulInt(int *A, int *B, int *C, int m, int n, int k) {

    // Allocate device memory:
    int *gpu_A;
    int *gpu_B;
    int *gpu_C;

    cudaMalloc((void **) &gpu_A, sizeof(int) * m * n);
    cudaMemcpy(gpu_A, A, sizeof(int) * m * n, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &gpu_B, sizeof(int) * n * k);
    cudaMemcpy(gpu_B, B, sizeof(int) * n * k, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &gpu_C, sizeof(int) * m * k);

    int BLOCK_SIZE = 16;
    // Blocks & grids:
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(grid_cols, grid_rows);

    // Call the kernel:
    vecmulInt<<<grid, blocks>>>(gpu_A, gpu_B, gpu_C, m, n, k);

    // Get the result Matrix:
    cudaMemcpy(C, gpu_C, sizeof(int) * m * k, cudaMemcpyDeviceToHost);

    //Free device matrices
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
}

void maxmulDouble(double *A, double *B, double *C, int m, int n, int k) {

    // Allocate device memory:
    double *gpu_A;
    double *gpu_B;
    double *gpu_C;

    // cout << "------------------\n";
    // cout << "A = {";
    //for (int i = 0; i < m * n; ++i) cout << A[i] << (i != m * n - 1 ? ", " : "}\n");
    cudaMalloc((void **) &gpu_A, sizeof(double) * m * n);
    cudaMemcpy(gpu_A, A, sizeof(double) * m * n, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &gpu_B, sizeof(double) * n * k);
    cudaMemcpy(gpu_B, B, sizeof(double) * n * k, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &gpu_C, sizeof(double) * m * k);

    int BLOCK_SIZE = 32;
    // Blocks & grids:
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(grid_cols, grid_rows);

    // Call the kernel:
    vecmulDouble<<<grid, blocks>>>(gpu_A, gpu_B, gpu_C, m, n, k);

    // Get the result Matrix:
    cudaMemcpy(C, gpu_C, sizeof(double) * m * k, cudaMemcpyDeviceToHost);
    // cout << "C = {";
    // for (int i = 0; i < m * k; ++i) cout << C[i] << (i != m * k - 1 ? ", " : "}\n");
    //Free device matrices
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
}