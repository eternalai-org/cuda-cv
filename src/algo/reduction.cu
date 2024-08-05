#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cstring>

#include <fixedlonglong32x32.cuh>
#include <kernels.cuh>
#include <operations.cuh>

long long sumReduction_impl(long long* d_gpu, int n, bool& error)
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
        res = sumReduction_impl(blockSum, grid_sz, error);
    }
    else
    {
        cudaMemcpy(&res, blockSum, sizeof(long long), cudaMemcpyDeviceToHost);
    }

    cudaFree(blockSum);
    return res;
}



long long maxReduction_impl(long long* d_gpu, int n, bool& error)
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
        res = maxReduction_impl(blockMax, grid_sz, error);
    }
    else
    {
        cudaMemcpy(&res, blockMax, sizeof(long long), cudaMemcpyDeviceToHost);
    }

    cudaFree(blockMax);
    return res;
}



long long minReduction_impl(long long* d_gpu, int n, bool& error)
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
        res = minReduction_impl(blockMin, grid_sz, error);
    }
    else
    {
        cudaMemcpy(&res, blockMin, sizeof(long long), cudaMemcpyDeviceToHost);
    }

    cudaFree(blockMin);
    return res;
}

long long __sumReduction(long long* inp, int n, bool& error)
{
    long long* gpu; 
    cudaMalloc(&gpu, n * sizeof(long long));
    cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice);
    long long res = sumReduction_impl(gpu, n, error);
    cudaFree(gpu);
    return res;
}

long long __avgReduction(long long* inp, int n, bool& error)
{
    return FixedLongLong::div(__sumReduction(inp, n, error), (1ll * n) << 32);
}

long long __maxReduction(long long* inp, int n, bool& error)
{
    long long* gpu; 
    cudaMalloc(&gpu, n * sizeof(long long));
    cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice);
    long long res = maxReduction_impl(gpu, n, error);
    cudaFree(gpu);
    return res;
}

long long __minReduction(long long* inp, int n, bool& error)
{
    long long* gpu; 
    cudaMalloc(&gpu, n * sizeof(long long));
    cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice);
    long long res = minReduction_impl(gpu, n, error);
    cudaFree(gpu);
    return res;
}

long long __meanReduction(long long* inp, int n, bool& error)
{
    return FixedLongLong::div(__sumReduction(inp, n, error), (1LL * n) << 32);
}

long long __stdReduction(long long* inp, int n, bool& error)
{
    long long mean = __meanReduction(inp, n, error);
    return 0;
}

void __maxMinScale(long long* inp, long long* out, int n, bool& error)
{
    long long min = __minReduction(inp, n, error);
    long long max = __maxReduction(inp, n, error);

}

void __zScore(long long* inp, long long* out, long long eps, int n, bool& error)
{
    long long mean = __meanReduction(inp, n, error); 
    long long std = __stdReduction(inp, n, error);
}
