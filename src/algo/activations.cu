#include <fixedlonglong32x32.cuh>
#include <operations.cuh>

// softmax interface
void __softmaxFixedLongLong(long long *A, long long* B, int m, uint8_t* error) 
{
    long long *gpu_a, *gpu_b, *buffer_tmp;

    
    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu_a, sizeof(long long)*m)))
    {
        cudaFree(gpu_a);
        return;
    }

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu_b, sizeof(long long)*m)))
    {
        cudaFree(gpu_b);
        return;
    }
    
    buffer_tmp = new long long[m];

    const int BLOCK_SIZE = 256;
    const int BLOCKS = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (*error = cuda_fmt_error(cudaMemcpy(gpu_a, A, sizeof(long long)*m, cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu_a);
        delete[] buffer_tmp;
        return;
    }

    const int sqrt_m = sqrt(m);
    const int BUCKETS = (m + sqrt_m - 1) / sqrt_m;
    arrayExp_kernel<<<BLOCKS, BLOCK_SIZE>>>(gpu_a, gpu_b, m);
    arraySum_kernel<<<BUCKETS, 1>>>(gpu_b, gpu_a, m);
    
    long long sumExp = 0;
    
    if (*error = cuda_fmt_error(cudaMemcpy(buffer_tmp, gpu_a, sizeof(long long)*BUCKETS, cudaMemcpyDeviceToHost)))
    {
        cudaFree(gpu_a);
        delete[] buffer_tmp;
        return;
    }

    for (int i = 0; i < BUCKETS; ++i)
    {
        sumExp += buffer_tmp[i];
    }

    softmaxImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu_b, gpu_a, m, sumExp);
    if (*error = cuda_fmt_error(cudaMemcpy(B, gpu_a, sizeof(long long)*m, cudaMemcpyDeviceToHost)))
    {
        cudaFree(gpu_a);
        delete[] buffer_tmp;
        return;
    }
    cudaFree(gpu_a), cudaFree(gpu_b);
    delete[] buffer_tmp;
}

// sigmoid interface
void __sigmoidFixedLongLong(long long *A, long long* B, int m, uint8_t* error) 
{  
    long long *gpu_a, *gpu_b;

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu_a, sizeof(long long)*m)))
    {
        cudaFree(gpu_a);
        return;
    }

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu_b, sizeof(long long)*m)))
    {
        cudaFree(gpu_b);
        return;
    }

    const int BLOCK_SIZE = 256;
    const int BLOCKS = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
 
    if (*error = cuda_fmt_error(cudaMemcpy(gpu_a, A, sizeof(long long)*m, cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu_a);
        return;
    }
    sigmoidImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu_a, gpu_b, m);

    if (*error = cuda_fmt_error(cudaMemcpy(B, gpu_b, sizeof(long long)*m, cudaMemcpyDeviceToHost)))
    {
        cudaFree(gpu_b);
        return;
    }
    cudaFree(gpu_a), cudaFree(gpu_b);
}

// tanh interface
void __tanhFixedLongLong(long long *A, long long *B, int m, uint8_t* error) 
{
    long long *gpu_a, *gpu_b;

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu_a, sizeof(long long)*m)))
    {
        cudaFree(gpu_a);
        return;
    }

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu_b, sizeof(long long)*m)))
    {
        cudaFree(gpu_b);
        return;
    }

    const int BLOCK_SIZE = 256;
    const int BLOCKS = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (*error = cuda_fmt_error(cudaMemcpy(gpu_a, A, sizeof(long long)*m, cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu_a);
        return;
    }
    tanhImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu_a, gpu_b, m);

    if (*error = cuda_fmt_error(cudaMemcpy(B, gpu_b, sizeof(long long)*m, cudaMemcpyDeviceToHost)))
    {
        cudaFree(gpu_b);
        return;
    }
    cudaFree(gpu_a), cudaFree(gpu_b);
}

// relu interface
void __reluFixedLongLong(long long *A, long long *B, int m, uint8_t* error) 
{
    long long *gpu_a, *gpu_b;

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu_a, sizeof(long long)*m)))
    {
        cudaFree(gpu_a);
        return;
    }

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu_b, sizeof(long long)*m)))
    {
        cudaFree(gpu_b);
        return;
    }

    const int BLOCK_SIZE = 256;
    const int BLOCKS = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (*error = cuda_fmt_error(cudaMemcpy(gpu_a, A, sizeof(long long)*m, cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu_a);
        return;
    }
    reluImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu_a, gpu_b, m);

    if (*error = cuda_fmt_error(cudaMemcpy(B, gpu_b, sizeof(long long)*m, cudaMemcpyDeviceToHost)))
    {
        cudaFree(gpu_b);
        return;
    }
    cudaFree(gpu_a), cudaFree(gpu_b);
}

// relu interface
void __relu3DFixedLongLong(long long *A, long long *B, int h, int w, int c, uint8_t* error) 
{
    long long* gpu;
    const int N = h * w * c;

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu, sizeof(long long) * N * 2)))
    {
        cudaFree(gpu);
        return;
    }

    if (*error = cuda_fmt_error(cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu);
        return;
    }
    
    const dim3 BLOCK_SIZE(256);
    const dim3 BLOCKS((N + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x);
    reluImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu, gpu + N, N);

    if (*error = cuda_fmt_error(cudaMemcpy(B, gpu + N, sizeof(long long) * N, cudaMemcpyDeviceToHost)))
    {
        cudaFree(gpu);
        return;
    }
    cudaFree(gpu);
}

// relu interface
void __sigmoid3DFixedLongLong(long long *A, long long *B, int h, int w, int c, uint8_t* error) 
{
    long long* gpu;
    const int N = h * w * c;

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu, sizeof(long long) * N * 2)))
    {
        cudaFree(gpu);
        return;
    }

    if (*error = cuda_fmt_error(cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu);
        return;
    }

    const dim3 BLOCK_SIZE(256);
    const dim3 BLOCKS((N + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x);
    sigmoidImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu, gpu + N, N);

    if (*error = cuda_fmt_error(cudaMemcpy(B, gpu + N, sizeof(long long) * N, cudaMemcpyDeviceToHost)))
    {
        cudaFree(gpu);
        return;
    }
    cudaFree(gpu);
}


// relu interface
void __tanh3DFixedLongLong(long long *A, long long *B, int h, int w, int c, uint8_t* error) 
{
    long long* gpu;
    const int N = h * w * c;

    if (*error = cuda_fmt_error(cudaMalloc((void**)&gpu, sizeof(long long) * N * 2)))
    {
        cudaFree(gpu);
        return;
    }

    if (*error = cuda_fmt_error(cudaMemcpy(gpu, A, sizeof(long long) * N, cudaMemcpyHostToDevice)))
    {
        cudaFree(gpu);
        return;
    }

    const dim3 BLOCK_SIZE(256);
    const dim3 BLOCKS((N + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x);
    tanhImplFixedLongLong<<<BLOCKS, BLOCK_SIZE>>>(gpu, gpu + N, N);

    if (*error = cuda_fmt_error(cudaMemcpy(B, gpu + N, sizeof(long long) * N, cudaMemcpyDeviceToHost)))
    {
        cudaFree(gpu);
        return;
    }
    cudaFree(gpu);
}

void __softmax2DFixedLongLong(long long* A, long long* B, int h, int w, int c, uint8_t* error)
{
    memset(B, 0, sizeof(long long) * h * w * c);
}