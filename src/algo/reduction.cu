#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cstring>

#include <fixedlonglong32x32.cuh>
#include <operations.h>
#include <kernels.cuh>

long long sumReduction_impl(long long* d_gpu, int n)
{
    long long res = 0;
    int block_sz = 512;
    int block_sz2 = block_sz * 2;
    int grid_sz = (n + block_sz2 - 1) / block_sz2;

    long long* blockSum;
    cudaMalloc(&blockSum, grid_sz * sizeof(long long));
    cudaMemset(blockSum, 0, grid_sz * sizeof(long long));

    sumReduction_kernel<<<grid_sz, block_sz, block_sz2 * sizeof(long long)>>>(d_gpu, blockSum, n);

    if (grid_sz > 1)
    {
        res = sumReduction_impl(blockSum, grid_sz);
    }
    else
    {
        cudaMemcpy(&res, blockSum, sizeof(long long), cudaMemcpyDeviceToHost);
    }

    cudaFree(blockSum);
    return res;
}



long long maxReduction_impl(long long* d_gpu, int n)
{
    long long res = 0;
    int block_sz = 1024;
    int block_sz2 = block_sz * 2;
    int grid_sz = (n + block_sz2 - 1) / block_sz2;

    long long* blockMax;
    cudaMalloc(&blockMax, grid_sz * sizeof(long long));
    cudaMemset(blockMax, 0, grid_sz * sizeof(long long));

    maxReduction_kernel<<<grid_sz, block_sz, block_sz2 * sizeof(long long)>>>(d_gpu, blockMax, n);

    if (grid_sz > 1)
    {
        res = maxReduction_impl(blockMax, grid_sz);
    }
    else
    {
        cudaMemcpy(&res, blockMax, sizeof(long long), cudaMemcpyDeviceToHost);
    }

    cudaFree(blockMax);
    return res;
}



long long minReduction_impl(long long* d_gpu, int n)
{
    long long res = 0;
    int block_sz = 512;
    int block_sz2 = block_sz * 2;
    int grid_sz = (n + block_sz2 - 1) / block_sz2;

    long long* blockMin;
    cudaMalloc(&blockMin, grid_sz * sizeof(long long));
    cudaMemset(blockMin, 0, grid_sz * sizeof(long long));

    sumReduction_kernel<<<grid_sz, block_sz, block_sz2 * sizeof(long long)>>>(d_gpu, blockMin, n);

    if (grid_sz > 1)
    {
        res = minReduction_impl(blockMin, grid_sz);
    }
    else
    {
        cudaMemcpy(&res, blockMin, sizeof(long long), cudaMemcpyDeviceToHost);
    }

    cudaFree(blockMin);
    return res;
}

long long sumReduction(long long* inp, int n)
{
    long long* gpu; 
    cudaMalloc(&gpu, n * sizeof(long long));
    cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice);
    long long res = sumReduction_impl(gpu, n);
    cudaFree(gpu);
    return res;
}

long long avgReduction(long long* inp, int n)
{
    return FixedLongLong::div(sumReduction(inp, n), (1ll * n) << 32);
}

long long maxReduction(long long* inp, int n)
{
    long long* gpu; 
    cudaMalloc(&gpu, n * sizeof(long long));
    cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice);
    long long res = maxReduction_impl(gpu, n);
    cudaFree(gpu);
    return res;
}

long long minReduction(long long* inp, int n)
{
    long long* gpu; 
    cudaMalloc(&gpu, n * sizeof(long long));
    cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice);
    long long res = minReduction_impl(gpu, n);
    cudaFree(gpu);
    return res;
}

long long meanReduction(long long* inp, int n)
{
    return FixedLongLong::div(sumReduction(inp, n), (1LL * n) << 32);
}

long long stdReduction(long long* inp, int n)
{
    long long mean = meanReduction(inp, n);
    return 0;
}

void maxMinScale(long long* inp, long long* out, int n)
{
    long long min = minReduction(inp, n);
    long long max = maxReduction(inp, n);

}

void zScore(long long* inp, long long* out, long long eps, int n)
{
    long long mean = meanReduction(inp, n); 
    long long std = stdReduction(inp, n);
}
