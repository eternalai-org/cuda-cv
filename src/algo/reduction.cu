#include <operations.cuh>

long long sumReduction_impl(long long* d_gpu, int n, uint8_t* error)
{
    long long res = 0;
    int block_sz = 512;
    int block_sz2 = block_sz * 2;
    int grid_sz = (n + block_sz2 - 1) / block_sz2;

    long long* blockSum;
    cudaMalloc(&blockSum, grid_sz * sizeof(long long));
    cudaMemset(blockSum, 0, grid_sz * sizeof(long long));

    sumReduction_kernel<<<grid_sz, block_sz, block_sz2 * sizeof(long long)>>>(d_gpu, blockSum, n);
    if (*error = cuda_fmt_error(cudaGetLastError()))
    {
        cudaFree(blockSum);
        return 0;
    }

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



long long maxReduction_impl(long long* d_gpu, int n, uint8_t* error)
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



long long minReduction_impl(long long* d_gpu, int n, uint8_t* error)
{
    long long res = 0;
    int block_sz = 512;
    int block_sz2 = block_sz * 2;
    int grid_sz = (n + block_sz2 - 1) / block_sz2;

    long long* blockMin;
    cudaMalloc(&blockMin, grid_sz * sizeof(long long));
    cudaMemset(blockMin, 0, grid_sz * sizeof(long long));

    minReduction_kernel<<<grid_sz, block_sz, block_sz2 * sizeof(long long)>>>(d_gpu, blockMin, n);

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

void channelWiseSumReduction_impl(long long* d_gpu, long long* d_out, int n, int c, uint8_t* error)
{
    long long* blockSum;

    const int block_size = 512;
    const int block_size_2 = block_size * 2;
    const int blocks = (n + block_size_2 - 1) / block_size_2;

    cudaMalloc(&blockSum, blocks * c * sizeof(long long));

    sumReductionV2_kernel<<<
        dim3(blocks, 1, c), 
        dim3(block_size, 1, 1), 
        sizeof(long long) * block_size_2>>>(d_gpu, blockSum, n, c);

    if (blocks > 1)
    {
        channelWiseSumReduction_impl(blockSum, d_out, blocks, c, error);
    }
    else
    {
        cudaMemcpy(d_out, blockSum, c * sizeof(long long), cudaMemcpyDeviceToDevice);
    }

    cudaFree(blockSum);
}

long long __sumReduction(long long* inp, int n, uint8_t* error)
{
    long long* gpu; 
    cudaMalloc(&gpu, n * sizeof(long long));
    cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice);
    long long res = sumReduction_impl(gpu, n, error);
    cudaFree(gpu);
    return res;
}

long long __avgReduction(long long* inp, int n, uint8_t* error)
{
    return FixedLongLong::div(__sumReduction(inp, n, error), (1ll * n) << 32);
}

long long __maxReduction(long long* inp, int n, uint8_t* error)
{
    long long* gpu; 
    cudaMalloc(&gpu, n * sizeof(long long));
    cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice);
    long long res = maxReduction_impl(gpu, n, error);
    cudaFree(gpu);
    return res;
}

long long __minReduction(long long* inp, int n, uint8_t* error)
{
    long long* gpu; 
    cudaMalloc(&gpu, n * sizeof(long long));
    cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice);
    long long res = minReduction_impl(gpu, n, error);
    cudaFree(gpu);
    return res;
}

long long __meanReduction(long long* inp, int n, uint8_t* error)
{
    return FixedLongLong::div(__sumReduction(inp, n, error), (1LL * n) << 32);
}

long long __stdReduction(long long* inp, int n, uint8_t* error)
{
    long long mean = __meanReduction(inp, n, error);
    return 0;
}

void __maxMinScale(long long* inp, long long* out, int n, uint8_t* error)
{
    long long min = __minReduction(inp, n, error);
    long long max = __maxReduction(inp, n, error);

}

void __zScore(long long* inp, long long* out, long long eps, int n, uint8_t* error)
{
    long long mean = __meanReduction(inp, n, error); 
    long long std = __stdReduction(inp, n, error);
}

void __channelWiseSumReduction(long long* inp, long long* out, int n, int c, uint8_t* error)
{
    long long* d_gpu = nullptr;
    cudaMalloc(&d_gpu, (n * c + c) * sizeof(long long));
    cudaMemcpy(d_gpu + c, inp, n * c * sizeof(long long), cudaMemcpyHostToDevice);
    channelWiseSumReduction_impl(d_gpu + c, d_gpu, n, c, error);
    cudaMemcpy(out, d_gpu, c * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaFree(d_gpu);
}