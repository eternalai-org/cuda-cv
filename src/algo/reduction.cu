#include <operations.cuh>

long long sumReduction_impl(long long* d_gpu, int n, uint8_t* error)
{
    long long res = 0;
    int block_sz = 512;
    int block_sz2 = block_sz * 2;
    int grid_sz = (n + block_sz2 - 1) / block_sz2;

    long long* blockSum;

    if (*error = cuda_fmt_error(cudaMalloc(&blockSum, grid_sz * sizeof(long long))))
    {
        cudaFree(blockSum);
        return 0;
    }

    if (*error = cuda_fmt_error(cudaMemset(blockSum, 0, grid_sz * sizeof(long long))))
    {
        cudaFree(blockSum);
        return 0;
    }

    sumReduction_kernel<<<grid_sz, block_sz, block_sz2 * sizeof(long long)>>>(d_gpu, blockSum, n);


    if (grid_sz > 1)
    {
        res = sumReduction_impl(blockSum, grid_sz, error);
    }
    else
    {
        if (*error = cuda_fmt_error(cudaMemcpy(&res, blockSum, sizeof(long long), cudaMemcpyDeviceToHost)))
        {
            cudaFree(blockSum);
            return 0;
        }
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

    if (*error = cuda_fmt_error(cudaMalloc(&blockMax, grid_sz * sizeof(long long))))
    {
        cudaFree(blockMax);
        return 0;
    }

    if (*error = cuda_fmt_error(cudaMemset(blockMax, 0, grid_sz * sizeof(long long))))
    {
        cudaFree(blockMax);
        return 0;
    }

    maxReduction_kernel<<<grid_sz, block_sz, block_sz2 * sizeof(long long)>>>(d_gpu, blockMax, n);

    if (grid_sz > 1)
    {
        res = maxReduction_impl(blockMax, grid_sz, error);
    }
    else
    {
        if (*error = cuda_fmt_error(cudaMemcpy(&res, blockMax, sizeof(long long), cudaMemcpyDeviceToHost)))
        {
            cudaFree(blockMax);
            return 0;
        }
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

    if (*error = cuda_fmt_error(cudaMalloc(&blockMin, grid_sz * sizeof(long long))))
    {
        cudaFree(blockMin);
        return 0;
    }

    if (*error = cuda_fmt_error(cudaMemset(blockMin, 0, grid_sz * sizeof(long long))))
    {
        cudaFree(blockMin);
        return 0;
    }

    minReduction_kernel<<<grid_sz, block_sz, block_sz2 * sizeof(long long)>>>(d_gpu, blockMin, n);

    if (grid_sz > 1)
    {
        res = minReduction_impl(blockMin, grid_sz, error);
    }
    else
    {
        if (*error = cuda_fmt_error(cudaMemcpy(&res, blockMin, sizeof(long long), cudaMemcpyDeviceToHost)))
        {
            cudaFree(blockMin);
            return 0;
        }
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

    ;
    if (*error = cuda_fmt_error(cudaMalloc(&blockSum, blocks * c * sizeof(long long))))
    {
        cudaFree(blockSum);
        return;
    }

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
        if (*error = cuda_fmt_error(cudaMemcpy(d_out, blockSum, c * sizeof(long long), cudaMemcpyDeviceToDevice)))
        {
            cudaFree(blockSum);
            return;
        }
    }

    cudaFree(blockSum);
}

long long __sumReduction(long long* inp, int n, uint8_t* error)
{
    long long* gpu; 
    if (*error = cuda_fmt_error(cudaMalloc(&gpu, n * sizeof(long long))))
    {
        cudaFree(gpu);
        return 0;
    }

    if (*error = cuda_fmt_error(cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu);
        return 0;
    }
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

    if (*error = cuda_fmt_error(cudaMalloc(&gpu, n * sizeof(long long))))
    {
        cudaFree(gpu);
        return 0;
    }

    if (*error = cuda_fmt_error(cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu);
        return 0;
    }
    long long res = maxReduction_impl(gpu, n, error);
    cudaFree(gpu);
    return res;
}

long long __minReduction(long long* inp, int n, uint8_t* error)
{
    long long* gpu; 

    if (*error = cuda_fmt_error(cudaMalloc(&gpu, n * sizeof(long long))))
    {
        cudaFree(gpu);
        return 0;
    }

    if (*error = cuda_fmt_error(cudaMemcpy(gpu, inp, n * sizeof(long long), cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu);
        return 0;
    }
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
    
    if (*error = cuda_fmt_error(cudaMalloc(&d_gpu, (n * c + c) * sizeof(long long))))
    {
        cudaFree(d_gpu);
        return;
    }

    if (*error = cuda_fmt_error(cudaMemcpy(d_gpu + c, inp, n * c * sizeof(long long), cudaMemcpyHostToDevice)))
    {
        cudaFree(d_gpu);
        return;
    }
    
    channelWiseSumReduction_impl(d_gpu + c, d_gpu, n, c, error);

    if (*error = cuda_fmt_error(cudaMemcpy(out, d_gpu, c * sizeof(long long), cudaMemcpyDeviceToHost)))
    {
        cudaFree(d_gpu);
        return;
    }
    
    cudaFree(d_gpu);
}