#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <assert.h>

#include <fixedlonglong32x32.cuh>
#include <functional.h>
#include <kernels.cuh>
#include <utilities.cuh>

// softmax interface
void softmaxFixedLongLong(long long *A, long long* B, int m) 
{
    long long *gpu_a, *gpu_b, *buffer_tmp;

    cudaMalloc((void**)&gpu_a, sizeof(long long)*m);
    cudaMalloc((void**)&gpu_b, sizeof(long long)*m);
    buffer_tmp = new long long[m];

    const int BLOCK_SIZE = 256;
    const int BLOCKS = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemcpy(gpu_a, A, sizeof(long long)*m, cudaMemcpyHostToDevice);

    const int sqrt_m = sqrt(m);
    const int BUCKETS = (m + sqrt_m - 1) / sqrt_m;
    arrayExp_kernel<<<BLOCKS, BLOCK_SIZE>>>(gpu_a, gpu_b, m);
    arraySum_kernel<<<BUCKETS, 1>>>(gpu_b, gpu_a, m);
    
    long long sumExp = 0;
    cudaMemcpy(buffer_tmp, gpu_a, sizeof(long long)*BUCKETS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < BUCKETS; ++i)
    {
        sumExp += buffer_tmp[i];
    }

    softmaxImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu_b, gpu_a, m, sumExp);
    cudaMemcpy(B, gpu_a, sizeof(long long)*m, cudaMemcpyDeviceToHost);
    cudaFree(gpu_a), cudaFree(gpu_b);
    delete[] buffer_tmp;
}

// sigmoid interface
void sigmoidFixedLongLong(long long *A, long long* B, int m) 
{  
    long long *gpu_a, *gpu_b;
    
    cudaMalloc((void**)&gpu_a, sizeof(long long)*m);
    cudaMalloc((void**)&gpu_b, sizeof(long long)*m);

    const int BLOCK_SIZE = 256;
    const int BLOCKS = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemcpy(gpu_a, A, sizeof(long long)*m, cudaMemcpyHostToDevice);
    sigmoidImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu_a, gpu_b, m);

    cudaMemcpy(B, gpu_b, sizeof(long long)*m, cudaMemcpyDeviceToHost);
    cudaFree(gpu_a), cudaFree(gpu_b);
}

// tanh interface
void tanhFixedLongLong(long long *A, long long *B, int m) 
{
    long long *gpu_a, *gpu_b;
    
    cudaMalloc((void**)&gpu_a, sizeof(long long)*m);
    cudaMalloc((void**)&gpu_b, sizeof(long long)*m);

    const int BLOCK_SIZE = 256;
    const int BLOCKS = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemcpy(gpu_a, A, sizeof(long long)*m, cudaMemcpyHostToDevice);
    tanhImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu_a, gpu_b, m);

    cudaMemcpy(B, gpu_b, sizeof(long long)*m, cudaMemcpyDeviceToHost);
    cudaFree(gpu_a), cudaFree(gpu_b);
}

// relu interface
void reluFixedLongLong(long long *A, long long *B, int m) 
{
    long long *gpu_a, *gpu_b;
    
    cudaMalloc((void**)&gpu_a, sizeof(long long)*m);
    cudaMalloc((void**)&gpu_b, sizeof(long long)*m);

    const int BLOCK_SIZE = 256;
    const int BLOCKS = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemcpy(gpu_a, A, sizeof(long long)*m, cudaMemcpyHostToDevice);
    reluImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu_a, gpu_b, m);

    cudaMemcpy(B, gpu_b, sizeof(long long)*m, cudaMemcpyDeviceToHost);
    cudaFree(gpu_a), cudaFree(gpu_b);
}

// relu interface
void relu3DFixedLongLong(long long *A, long long *B, int h, int w, int c) 
{
    long long* gpu;
    const int N = h * w * c;

    cudaMalloc((void**)&gpu, sizeof(long long) * N * 2);
    cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice);

    const dim3 BLOCK_SIZE(256);
    const dim3 BLOCKS((N + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x);
    reluImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu, gpu + N, N);

    cudaMemcpy(B, gpu + N, sizeof(long long) * N, cudaMemcpyDeviceToHost);
    cudaFree(gpu);
}

// relu interface
void sigmoid3DFixedLongLong(long long *A, long long *B, int h, int w, int c) 
{
    long long* gpu;
    const int N = h * w * c;

    cudaMalloc((void**)&gpu, sizeof(long long) * N * 2);
    cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice);

    const dim3 BLOCK_SIZE(256);
    const dim3 BLOCKS((N + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x);
    sigmoidImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu, gpu + N, N);

    cudaMemcpy(B, gpu + N, sizeof(long long) * N, cudaMemcpyDeviceToHost);
    cudaFree(gpu);
}


// relu interface
void tanh3DFixedLongLong(long long *A, long long *B, int h, int w, int c) 
{
    long long* gpu;
    const int N = h * w * c;

    cudaMalloc((void**)&gpu, sizeof(long long) * N * 2);
    cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice);

    const dim3 BLOCK_SIZE(256);
    const dim3 BLOCKS((N + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x);
    tanhImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu, gpu + N, N);

    cudaMemcpy(B, gpu + N, sizeof(long long) * N, cudaMemcpyDeviceToHost);
    cudaFree(gpu);
}

void softmax2DFixedLongLong(long long* A, long long* B, int h, int w, int c)
{
    memset(B, 0, sizeof(long long) * h * w * c);
}