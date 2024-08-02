#include <chrono>
#include <fstream>
#include <iostream>
#include <operations.h>
#include <helpers.cuh>
#include <vector>

using namespace std;

void benchmark_conv2d()
{
    // reset random seed
    srand(time(nullptr));

    int filters = 512, kh = 1, kw = 1, c = 1024;
    long long* kernel = new long long[filters * kh * kw * c];
    long long* bias = new long long[filters];

    int padding = 0;
    int stride_h = 1, stride_w = 1;

    for (int i = 0; i < filters * kh * kw * c; ++i)
    {
        kernel[i] = rand() % 100;
        kernel[i] <<= 32;
    }

    for (int i = 0; i < filters; ++i)
    {
        bias[i] = rand() % 100;
        bias[i] <<= 32;
    }

    long long params = 512 * 1 * 1 * 1024 + 512;
    std::cerr << "Benchmarking Conv2D 512 filters 1 x 1; ; Params: " << 1.0f * params / 1e6 << "M" << '\n';
    
    for (int i = 1; i < 10; ++i)
    {
        int h = 1 << i, w = 1 << i;
        long long inpSize = h * w * c;
        std::cerr << "Input shape: " << h << " x " << w << " x " << c << "; Size: " << 1.0f * inpSize / 1e6 << "M" << '\n';

        long long* input = new long long[h * w * c];
        long long* output = new long long[h * w * filters];
        
        for (int i = 0; i < h * w * c; ++i)
        {
            input[i] = rand() % 100;
            input[i] <<= 32;
        }

        auto start = std::chrono::high_resolution_clock::now();
        conv2dFixedLongLong(
            input, kernel, bias, output, kw, c, filters, 
            h, w, padding, stride_h, stride_w
        );
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cerr << "Time: " << elapsed.count() << " s" << '\n' << '\n';
    
        delete[] input;
        delete[] output;
    }

    delete[] kernel;
    delete[] bias;
}

int main(int argc, char** argv)
{
    long long ma[] = {0, 0, 0, 0, 0};
    long long mv[] = {1, 1, 1, 1, 1};
    long long gama[] = {1, 1, 1, 1, 1};
    long long beta[] = {0, 0, 0, 0, 0};

    long long inp[] = {1, 2, 3, 4, 5};

    for (int i = 0; i < 5; ++i)
    {
        ma[i] <<= 32;
        mv[i] <<= 32;
        gama[i] <<= 32;
        beta[i] <<= 32;
        inp[i] <<= 32;
    }

    long long out[5];
    batchNormalizeFixedLongLong(
        inp, out, ma, mv, gama, beta, 
        0, 1, 1, 5
    );
    std::cout << "Batch Normalization: " << '\n';
    printmat3d(out, 5, 1, 1);

    softmaxFixedLongLong(inp, out, 5);
    std::cout << "Softmax: " << '\n';
    printmat3d(out, 5, 1, 1);

    sigmoidFixedLongLong(inp, out, 5);
    std::cout << "Sigmoid: " << '\n';
    printmat3d(out, 5, 1, 1);

    tanhFixedLongLong(inp, out, 5);
    std::cout << "Tanh: " << '\n';
    printmat3d(out, 5, 1, 1);

    reluFixedLongLong(inp, out, 5);
    std::cout << "Relu: " << '\n';
    printmat3d(out, 5, 1, 1);

    relu3DFixedLongLong(inp, out, 1, 1, 5);
    std::cout << "Relu3D: " << '\n';
    printmat3d(out, 5, 1, 1);

    sigmoid3DFixedLongLong(inp, out, 1, 1, 5);
    std::cout << "Sigmoid3D: " << '\n';
    printmat3d(out, 5, 1, 1);

    tanh3DFixedLongLong(inp, out, 1, 1, 5);
    std::cout << "Tanh3D: " << '\n';
    printmat3d(out, 5, 1, 1);

    softmax2DFixedLongLong(inp, out, 1, 1, 5);
    std::cout << "Softmax2D: " << '\n';
    printmat3d(out, 5, 1, 1);
    
    long long shapes[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    concatenate(inp, out, shapes, 0, 3, 5);
    std::cout << "Concatenate: " << '\n';
    printmat3d(out, 5, 1, 1);

    std::cout << "Input: " << '\n';
    printmat3d(inp, 5, 1, 1);

    long long sum = sumReduction(inp, 5);
    std::cout << "Sum: " << 1.0f * sum / (1LL << 32) << '\n';

    long long max = maxReduction(inp, 5);
    std::cout << "Max: " << 1.0f * max / (1LL << 32) << '\n';

    long long min = minReduction(inp, 5);
    std::cout << "Min: " << 1.0f * min / (1LL << 32) << '\n';

    long long mean = meanReduction(inp, 5);
    std::cout << "Mean: " << 1.0f * mean / (1LL << 32) << '\n';

    long long std = stdReduction(inp, 5);
    std::cout << "Std: " << 1.0f * std / (1LL << 32) << '\n';

    long long avg = avgReduction(inp, 5);
    std::cout << "Avg: " << 1.0f * avg / (1LL << 32) << '\n';

    globalAvgPoolingFixedLongLong(inp, out, 1, 1, 5);
    std::cout << "Global Avg Pooling: " << '\n';
    printmat3d(out, 1, 1, 5);

    // 4 x 4
    long long inp44[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    for (int i = 0; i < 16; ++i)
    {
        inp44[i] <<= 32;
    }

    std::cout << "Input: " << '\n';
    printmat3d(inp44, 4, 4, 1);

    long long out22[4];
    maxPoolingFixedLongLong(inp44, out22, 4, 4, 1, 2, 2, 2, 1);
    std::cout << "Max Pooling: " << '\n';
    printmat3d(out22, 2, 2, 1);

    avgPoolingFixedLongLong(inp44, out22, 4, 4, 1, 2, 2, 2, 1);
    std::cout << "Avg Pooling: " << '\n';
    printmat3d(out22, 2, 2, 1);

    benchmark_conv2d();

    long long concatout[16];
    long long _shapes[] = {2, 4, 1, 2, 4, 1};

    std::cout << "Input 1: " << '\n';
    printmat3d(inp44, 2, 4, 1);
    
    std::cout << "Input 2: " << '\n';
    printmat3d(inp44 + 8, 2, 4, 1);

    concatenate(inp44, concatout, _shapes, 1, 3, 2);

    std::cout << "Concatenate: " << '\n';
    printmat3d(concatout, 2, 8, 1);

    std::cout << "Mat mul:" << '\n';

    maxmulFixedLongLong(inp44, inp44 + 8, out22, 4, 2, 4);

    printmat3d(out22, 2, 2, 1);

    return 0;
}
