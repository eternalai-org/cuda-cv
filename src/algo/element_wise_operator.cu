#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <assert.h>

#include <fixedlonglong32x32.cuh>
#include <operations.h>
#include <kernels.cuh>


////////////////////// kernels ///////////////////////// 


////////////////////// implementations ///////////////////////// 

void matAddLongLong(long long *A, long long *B, long long *C, int m, int n) {
    // Allocate device memory:
    long long *gpu;
    const int N = m * n;

    cudaMalloc((void **) &gpu, sizeof(long long) * N * 3);
    cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu + N, B, sizeof(long long) * N, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mat_add_fixed_longlong<<<GRID_SIZE, BLOCK_SIZE>>>(gpu, gpu + N, gpu + 2 * N, N);

    cudaMemcpy(C, gpu + 2 * N, sizeof(long long) * N, cudaMemcpyDeviceToHost);
    cudaFree(gpu);
}

void matSubLongLong(long long *A, long long *B, long long *C, int m, int n) {
    // Allocate device memory:
    long long *gpu;
    const int N = m * n;

    cudaMalloc((void **) &gpu, sizeof(long long) * N * 3);
    cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu + N, B, sizeof(long long) * N, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mat_sub_fixed_longlong<<<GRID_SIZE, BLOCK_SIZE>>>(gpu, gpu + N, gpu + 2 * N, N);

    cudaMemcpy(C, gpu + 2 * N, sizeof(long long) * N, cudaMemcpyDeviceToHost);
    cudaFree(gpu);
}

void matMulLongLong(long long *A, long long *B, long long *C, int m, int n) {
    // Allocate device memory:
    long long *gpu;
    const int N = m * n;

    cudaMalloc((void **) &gpu, sizeof(long long) * N * 3);
    cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu + N, B, sizeof(long long) * N, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mat_mul_fixed_longlong<<<GRID_SIZE, BLOCK_SIZE>>>(gpu, gpu + N, gpu + 2 * N, N);

    cudaMemcpy(C, gpu + 2 * N, sizeof(long long) * N, cudaMemcpyDeviceToHost);
    cudaFree(gpu);
}

void matDivLongLong(long long *A, long long *B, long long *C, int m, int n) {
    // Allocate device memory:
    long long *gpu;
    const int N = m * n;

    cudaMalloc((void **) &gpu, sizeof(long long) * N * 3);
    cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu + N, B, sizeof(long long) * N, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mat_div_fixed_longlong<<<GRID_SIZE, BLOCK_SIZE>>>(gpu, gpu + N, gpu + 2 * N, N);

    cudaMemcpy(C, gpu + 2 * N, sizeof(long long) * N, cudaMemcpyDeviceToHost);
    cudaFree(gpu);
}


void matSqrtLongLong(long long *A, long long *B, int m, int n) {
    // Allocate device memory:
    long long *gpu;
    const int N = m * n;

    cudaMalloc((void **) &gpu, sizeof(long long) * N * 2);
    cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mat_sqrt_fixed_longlong<<<GRID_SIZE, BLOCK_SIZE>>>(gpu, gpu + N, N);

    cudaMemcpy(B, gpu + N, sizeof(long long) * N, cudaMemcpyDeviceToHost);
    cudaFree(gpu);
}