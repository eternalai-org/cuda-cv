#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cstring>

#include <fixedlonglong32x32.cuh>
#include <functional.h>
#include <kernels.cuh>


////////////////////// implementation ///////////////////////// 


void globalAvgPoolingFixedLongLong_impl(
    long long* inp, long long* out,
    int h, int w, int in_channel
)
{
    const int block_sz = 512;
    const int block_sz2 = block_sz * 2;
    const int grid_sz_x = (h * w + block_sz2 - 1) / block_sz2;
    const int grid_sz_z = in_channel;

    long long* blockSum;
    cudaMalloc(&blockSum, sizeof(long long) * grid_sz_x * grid_sz_z);

    sumReductionV2_kernel<<<dim3(grid_sz_x, 1, grid_sz_z), block_sz, sizeof(long long) * block_sz2>>>(
        inp, blockSum, h * w, in_channel
    );

    if (grid_sz_x > 1)
    {
        globalAvgPoolingFixedLongLong_impl(
            blockSum, out, grid_sz_x, 1, in_channel
        );
    }
    else
    {
        cudaMemcpy(out, blockSum, in_channel * sizeof(long long), cudaMemcpyDeviceToDevice);
    }

    cudaFree(blockSum);    
}

void maxPoolingFixedLongLong(
    long long* inp, long long* out, // data io
    int h, int w, int in_channel, // in spatial size, in_channel
    int pool_size, int stride_h, int stride_w, // pooling size, stride 
    int padding // padding mode, one of 'valid': 0 or 'same': 1
)
{
    // only support for squared pool size, squared input

    if (w != h)
    {
        // not sure if it works
        return;
    }

    if (stride_h <= 0)
    {
        stride_h = pool_size;
    }

    if (stride_w <= 0)
    {
        stride_w = pool_size;
    }

    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;

    if (padding == 1)
    {
        int out_h = (h + stride_h - 1) / stride_h;
        int out_w = (w + stride_w - 1) / stride_w;

        int pad_h = max((out_h - 1) * stride_h + pool_size - h, 0);
        int pad_w = max((out_w - 1) * stride_w + pool_size - w, 0);
        
        pad_top = pad_h / 2;
        pad_bottom = pad_h - pad_top;
        
        pad_left = pad_w / 2;
        pad_right = pad_w - pad_left;
    }

    long long *d_gpu, *padded_inp;

    int out_w = (w + pad_left + pad_right - pool_size) / stride_w + 1;
    int out_h = (h + pad_top + pad_bottom - pool_size) / stride_h + 1;

    uint64_t inpFlatSize = (w + pad_left + pad_right) * (h + pad_top + pad_bottom) * in_channel;
    uint64_t outFlatSize = out_h * out_w * in_channel;
    uint64_t flatSize = inpFlatSize + outFlatSize;

    cudaMalloc(&d_gpu, flatSize * sizeof(long long));
    padded_inp = new long long[inpFlatSize];

    // printf("Input\n");
    // printmat3d(inp, h, w, in_channel);

    memset(padded_inp, 0x00, inpFlatSize * sizeof(long long));

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            memcpy(padded_inp + ((i + pad_top) * (w + pad_left + pad_right) + j + pad_left) * in_channel, inp + (i * w + j) * in_channel, in_channel << 3);
            // for (int k = 0; k < in_channel; ++k)
            // {
            //     padded_inp[((i + pad_top) * (w + pad_left + pad_right) + j + pad_left) * in_channel + k] = inp[(i * w + j) * in_channel + k];
            // }
        }
    }

    // printf("Padded input\n");
    // printmat3d(padded_inp, h + pad_bottom + pad_top, w + pad_left + pad_right, in_channel);

    cudaMemcpy(d_gpu, padded_inp, inpFlatSize * sizeof(long long), cudaMemcpyHostToDevice);
    delete[] padded_inp;

    const int thread_x = 32, thread_y=32, thread_z=1; 
    dim3 threads_per_block(thread_x, thread_y, thread_z);

    const dim3 block_per_grid(
        (out_w + threads_per_block.x - 1) / threads_per_block.x,
        (out_h + threads_per_block.y - 1) / threads_per_block.y,
        (in_channel + threads_per_block.z - 1) / threads_per_block.z
    );

    maxPoolingImplFixedLongLong_kernel<<<block_per_grid, threads_per_block>>>(
        d_gpu, d_gpu + inpFlatSize, 
        h + pad_bottom + pad_top, w + pad_left + pad_right, in_channel, 
        pool_size, stride_h, stride_w, 
        out_h, out_w, 
        pad_top, pad_bottom, pad_left, pad_right
    );

    cudaMemcpy(out, d_gpu + inpFlatSize, outFlatSize * sizeof(long long), cudaMemcpyDeviceToHost);

    // printf("Output\n");
    // printmat3d(out, out_h, out_w, in_channel);

    // printf("OutH %d\n", out_h);
    // printf("OutW %d\n", out_w);
    // printf("OutC %d\n", in_channel);

    cudaFree(d_gpu);
}

void avgPoolingFixedLongLong(
    long long* inp, long long* out,
    int h, int w, int in_channel,
    int pool_size, int stride_h, int stride_w,
    int padding
)
{
    // only support for squared pool size, squared input
    // only support for squared pool size, squared input

    if (w != h)
    {
        // not sure if it works
        return;
    }

    if (stride_h <= 0)
    {
        stride_h = pool_size;
    }

    if (stride_w <= 0)
    {
        stride_w = pool_size;
    }

    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;

    if (padding == 1)
    {
        const int _out_h = (h + stride_h - 1) / stride_h;
        const int _out_w = (w + stride_w - 1) / stride_w;

        const int pad_h = max((_out_h - 1) * stride_h + pool_size - h, 0);
        const int pad_w = max((_out_w - 1) * stride_w + pool_size - w, 0);
        
        pad_top = pad_h / 2;
        pad_bottom = pad_h - pad_top;
        
        pad_left = pad_w / 2;
        pad_right = pad_w - pad_left;
    }

    long long *d_gpu, *padded_inp;

    const int out_w = (w + pad_left + pad_right - pool_size) / stride_w + 1;
    const int out_h = (h + pad_top + pad_bottom - pool_size) / stride_h + 1;

    const uint64_t inpFlatSize = (w + pad_left + pad_right) * (h + pad_top + pad_bottom) * in_channel;
    const uint64_t outFlatSize = out_h * out_w * in_channel;
    const uint64_t cudaMemSize = inpFlatSize + outFlatSize;

    cudaMalloc(&d_gpu, cudaMemSize * sizeof(long long));
    padded_inp = new long long[inpFlatSize];

    memset(padded_inp, 0x00, inpFlatSize * sizeof(long long));

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            memcpy(padded_inp + ((i + pad_top) * (w + pad_left + pad_right) + j + pad_left) * in_channel, inp + (i * w + j) * in_channel, in_channel << 3);
            // for (int k = 0; k < in_channel; ++k)
            // {
            //     padded_inp[((i + pad_top) * (w + pad_left + pad_right) + j + pad_left) * in_channel + k] = inp[(i * w + j) * in_channel + k];
            // }
        }
    }

    cudaMemcpy(d_gpu, padded_inp, inpFlatSize * sizeof(long long), cudaMemcpyHostToDevice);
    delete[] padded_inp;

    const int thread_x = 32, thread_y=32, thread_z = 1;
    dim3 threads_per_block(thread_x, thread_y, thread_z);

    const dim3 block_per_grid(
        (out_w + threads_per_block.x - 1) / threads_per_block.x,
        (out_h + threads_per_block.y - 1) / threads_per_block.y,
        (in_channel + threads_per_block.z - 1) / threads_per_block.z
    );


    avgPoolingImplFixedLongLong_kernel<<<block_per_grid, threads_per_block>>>(
        d_gpu, d_gpu + inpFlatSize, 
        h + pad_top + pad_bottom, w + pad_left + pad_right, in_channel, 
        pool_size, stride_h, stride_w, 
        out_h, out_w,
        pad_top, pad_bottom, pad_left, pad_right
    );

    cudaMemcpy(out, d_gpu + inpFlatSize, outFlatSize * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaFree(d_gpu);
}

void globalAvgPoolingFixedLongLong(
    long long* inp, long long* out,
    int h, int w, int in_channel
)
{
    long long* gpu;
    cudaMalloc(&gpu, sizeof(long long) * (h * w * in_channel + in_channel));
    cudaMemcpy(gpu + in_channel, inp, h * w * in_channel * sizeof(long long), cudaMemcpyHostToDevice);
    globalAvgPoolingFixedLongLong_impl(gpu + in_channel, gpu, h, w, in_channel);
    cudaMemcpy(out, gpu, in_channel * sizeof(long long), cudaMemcpyDeviceToHost);

    // assume the number of channel is not too large at the moment
    for (int i = 0; i < in_channel; ++i)
    {
        out[i] = FixedLongLong::div(
            out[i], 
            FixedLongLong::mul(FixedLongLong::fromInt(h), FixedLongLong::fromInt(w))
        );
    }

    cudaFree(gpu);
}

void estimatePoolingOutputSize(
    int h, int w, int in_channel,
    int pool_size, int padding, 
    int stride_h, int stride_w,
    int& out_h, int& out_w
)
{
    if (stride_h <= 0)
    {
        stride_h = pool_size;
    }

    if (stride_w <= 0)
    {
        stride_w = pool_size;
    }

    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;

    if (padding == 1)
    {
        int out_h = (h + stride_h - 1) / stride_h;
        int out_w = (w + stride_w - 1) / stride_w;

        int pad_h = max((out_h - 1) * stride_h + pool_size - h, 0);
        int pad_w = max((out_w - 1) * stride_w + pool_size - w, 0);
        
        pad_top = pad_h / 2;
        pad_bottom = pad_h - pad_top;
        
        pad_left = pad_w / 2;
        pad_right = pad_w - pad_left;
    }

    out_w = (w + pad_left + pad_right - pool_size) / stride_w + 1;
    out_h = (h + pad_top + pad_bottom - pool_size) / stride_h + 1;
}