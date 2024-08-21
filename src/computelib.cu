#include <operations.cuh>
#include <computelib.h>
#include <tensor.h>
#include <memory.h>
#include <iostream>
#include <fstream> 
#include <bitset>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

uint8_t* conv2d_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 3 || pack.tensors[0].shape().size() != 3)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<int64_t>& params = pack.params;

    const std::vector<uint64_t>& inp = pack.tensors[0].shape(),
                                 kernel = pack.tensors[1].shape(),
                                 bias = pack.tensors[2].shape();

    // inp: [h, w, in_c]
    // kernel: [kh, kw, in_c, out_c]
    // bias: [out_c]
    if (kernel.size() != 4 || bias.size() != 1 || kernel[3] != bias[0] || kernel[2] != inp[2])
    {
        *_error = true;
        return nullptr;
    }

    uint32_t h_in = inp[0], w_in = inp[1], c_in = inp[2], c_out = kernel[3], h_out, w_out;
    uint32_t kh = kernel[0], kw = kernel[1], padding = params[2], stride_h = params[0], stride_w = params[1];
    estimateConvOutputSize(kh, c_in, c_out, h_in, w_in, padding, stride_h, stride_w, (int*) &h_out, (int*) &w_out);

    std::vector<uint64_t> out_shape = {h_out, w_out, c_out};
    int64_t* out = new int64_t[h_out * w_out * c_out];

    __conv2dFixedLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)pack.tensors[1].data(), 
        (long long*)pack.tensors[2].data(), 
        (long long*)out, 
        kw, c_in, c_out, 
        h_in, w_in, padding, 
        stride_h, stride_w, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(out_shape, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* maxpooling2d_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 1 || pack.tensors[0].shape().size() != 3)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<int64_t>& params = pack.params;
    const std::vector<uint64_t>& inp = pack.tensors[0].shape();

    if (params.size() != 5)
    {
        *_error = true;
        return nullptr;
    }

    uint32_t h_in = inp[0], w_in = inp[1], c_in = inp[2], h_out, w_out;
    uint32_t kh = params[0], kw = params[1], stride_h = params[2], stride_w = params[3], padding = params[4];

    estimatePoolingOutputSize(
        h_in, w_in, c_in, kh, padding, stride_h, stride_w, 
        (int*)&h_out, (int*)&w_out
    );

    std::vector<uint64_t> out_shape = {h_out, w_out, c_in};
    int64_t* out = new int64_t[h_out * w_out * c_in];

    __maxPoolingFixedLongLong(
        (long long*) pack.tensors[0].data(),
        (long long*) out, 
        h_in, w_in, c_in, kh, 
        stride_h, stride_w, padding, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(out_shape, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* avgpooling2d_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 1 || pack.tensors[0].shape().size() != 3)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<int64_t>& params = pack.params;
    const std::vector<uint64_t>& inp = pack.tensors[0].shape();

    if (params.size() != 5)
    {
        *_error = true;
        return nullptr;
    }

    uint32_t h_in = inp[0], w_in = inp[1], c_in = inp[2], h_out, w_out;
    uint32_t kh = params[0], kw = params[1], stride_h = params[2], stride_w = params[3], padding = params[4];

    estimatePoolingOutputSize(
        h_in, w_in, c_in, kh, padding, stride_h, stride_w, (int*)&h_out, (int*)&w_out
    );

    std::vector<uint64_t> out_shape = {h_out, w_out, c_in};
    int64_t* out = new int64_t[h_out * w_out * c_in];

    __avgPoolingFixedLongLong(
        (long long*)pack.tensors[0].data(),
        (long long*)out, h_in, w_in, c_in, kh, 
        stride_h, stride_w, padding, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(out_shape, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* matmul_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 2)
    {
        *_error = true;
        return nullptr;
    }

    uint32_t h1 = 1, 
        w1 = *pack.tensors[0].shape().rbegin(), 
        h2 = 1, 
        w2 = *pack.tensors[1].shape().rbegin();

    const auto& shape1 = pack.tensors[0].shape();
    const auto& shape2 = pack.tensors[1].shape();

    for (int i = 0; i < shape1.size() - 1; ++i) 
    {
        h1 *= shape1[i];
    }

    for (int i = 0; i < shape2.size() - 1; ++i) 
    {
        h2 *= shape2[i];
    }

    if (w1 != h2)
    {
        *_error = true;
        return nullptr;
    }

    std::vector<uint64_t> out_shape = {h1, w2};
    int64_t* out = new int64_t[h1 * w2];  

    __maxmulFixedLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)pack.tensors[1].data(), 
        (long long*)out, 
        h1, w1, w2, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(out_shape, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* elementwise_add_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 2)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& s1 = pack.tensors[0].shape(),
                                 s2 = pack.tensors[1].shape();

    int prod1 = std::accumulate(s1.begin(), s1.end(), 1, std::multiplies<int64_t>());
    int prod2 = std::accumulate(s2.begin(), s2.end(), 1, std::multiplies<int64_t>());

    if (prod1 != prod2)
    {
        *_error = true;
        return nullptr;
    }

    int64_t* out = new int64_t[prod1];
    __matAddLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)pack.tensors[1].data(), 
        (long long*)out, 
        1,
        prod1, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(s1, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* elementwise_mul_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 2)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& s1 = pack.tensors[0].shape(),
                                 s2 = pack.tensors[1].shape();

    int prod1 = std::accumulate(s1.begin(), s1.end(), 1, std::multiplies<int64_t>());
    int prod2 = std::accumulate(s2.begin(), s2.end(), 1, std::multiplies<int64_t>());

    if (prod1 != prod2)
    {
        *_error = true;
        return nullptr;
    }

    int64_t* out = new int64_t[prod1];
    __matMulLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)pack.tensors[1].data(), 
        (long long*)out, 
        1,
        prod1, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(s1, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* elementwise_sub_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 2)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& s1 = pack.tensors[0].shape(),
                                 s2 = pack.tensors[1].shape();

    int prod1 = std::accumulate(s1.begin(), s1.end(), 1, std::multiplies<int64_t>());
    int prod2 = std::accumulate(s2.begin(), s2.end(), 1, std::multiplies<int64_t>());

    if (prod1 != prod2)
    {
        *_error = true;
        return nullptr;
    }

    int64_t* out = new int64_t[prod1];
    __matSubLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)pack.tensors[1].data(), 
        (long long*)out, 
        prod1, 
        1,
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(s1, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* elementwise_div_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 2)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& s1 = pack.tensors[0].shape(),
                                 s2 = pack.tensors[1].shape();

    int prod1 = std::accumulate(s1.begin(), s1.end(), 1, std::multiplies<int64_t>());
    int prod2 = std::accumulate(s2.begin(), s2.end(), 1, std::multiplies<int64_t>());

    if (prod1 != prod2)
    {
        *_error = true;
        return nullptr;
    }

    int64_t* out = new int64_t[prod1];
    __matDivLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)pack.tensors[1].data(), 
        (long long*)out, 
        1,
        prod1, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(s1, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* transform_exp_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* transform_sqrt_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* batch_norm_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    // inp, ma, mv, gama, beta
    if (pack.tensors.size() != 5)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& inp = pack.tensors[0].shape(),
                                 ma = pack.tensors[1].shape(),
                                 mv = pack.tensors[2].shape(),
                                 gama = pack.tensors[3].shape(),
                                 beta = pack.tensors[4].shape();

    const std::vector<int64_t>& params = pack.params;

    const int h_in = inp[0], w_in = inp[1], c_in = inp[2];
    
    if (ma.size() != 1 || mv.size() != 1 || gama.size() != 1 || beta.size() != 1)
    {
        *_error = true;
        return nullptr;
    }

    if (params.size() == 0 || ma[0] != c_in || mv[0] != c_in || gama[0] != c_in || beta[0] != c_in)
    {
        *_error = true;
        return nullptr;
    }

    int64_t* out = new int64_t[h_in * w_in * c_in];
    __batchNormalizeFixedLongLong(
        (long long*)pack.tensors[0].data(), // inp
        (long long*)out, 
        (long long*)pack.tensors[1].data(), // ma
        (long long*)pack.tensors[2].data(), // mv
        (long long*)pack.tensors[3].data(), // gama
        (long long*)pack.tensors[4].data(), // beta 
        params[0], // epsilon
        h_in, w_in, c_in, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(inp, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* layer_norm_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* zscore_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* min_max_scale_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* concatenate_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    int n_tensors = pack.tensors.size();
    
    if (!n_tensors)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<int64_t>& params = pack.params;

    if (params.size() == 0)
    {
        *_error = true;
        return nullptr;
    }

    for (int i = 1; i < n_tensors; ++i)
    {
        if (pack.tensors[i].shape().size() != pack.tensors[0].shape().size())
        {
            *_error = true;
            return nullptr;
        }
    }

    int64_t** inp_tensors = new int64_t*[n_tensors];
    int64_t** shapes = new int64_t*[n_tensors];
    int common_dims = pack.tensors[0].shape().size();

    for (int i = 0; i < n_tensors; ++i)
    {
        inp_tensors[i] = pack.tensors[i].data();
        shapes[i] = new int64_t[common_dims];
        const auto& x = pack.tensors[i].shape();
        std::copy(x.begin(), x.end(), shapes[i]);
    }

    long long* out_shape = new long long[common_dims];

    if (!estimateConcatenate_dummy((long long**)shapes, params[0], common_dims, n_tensors, out_shape))
    {
        *_error = true;
        delete[] inp_tensors;

        for (int i = 0; i < n_tensors; ++i)
        {
            delete[] shapes[i];
        }

        delete[] shapes;
        delete[] out_shape;
        return nullptr;
    }
    
    int32_t prod = std::accumulate(out_shape, out_shape + common_dims, 1, std::multiplies<long long>());

    int64_t* out = new int64_t[prod];
    __concatenate_dummy(
        (long long**)inp_tensors, 
        (long long*)out, 
        (long long**)shapes, 
        params[0], 
        common_dims, 
        n_tensors,
        _error
    );

    if (*_error)
    {
        delete[] out;
        delete[] inp_tensors;

        for (int i = 0; i < n_tensors; ++i)
        {
            delete[] shapes[i];
        }

        delete[] shapes;
        delete[] out_shape;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(std::vector<uint64_t>(out_shape, out_shape + common_dims), out), 
        length_out
    );

    delete[] out;
    delete[] inp_tensors;

    for (int i = 0; i < n_tensors; ++i)
    {
        delete[] shapes[i];
    }

    delete[] shapes;
    delete[] out_shape;

    return out_bytes;
}

uint8_t* relu_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    int n_tensor = pack.tensors.size();
    
    if (n_tensor != 1)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& inp = pack.tensors[0].shape();
    uint64_t prod = 1;
    
    for (const int& x: inp)
    {
        prod *= x;
    }

    int64_t* out = new int64_t[prod];
    __reluFixedLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)out, 
        prod, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(inp, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* tanh_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    int n_tensor = pack.tensors.size();
    
    if (n_tensor != 1)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& inp = pack.tensors[0].shape();
    uint64_t prod = std::accumulate(inp.begin(), inp.end(), 1, std::multiplies<int64_t>());

    int64_t* out = new int64_t[prod];
    __tanhFixedLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)out, 
        prod, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(inp, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* sigmoid_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    int n_tensor = pack.tensors.size();
    
    if (n_tensor != 1)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& inp = pack.tensors[0].shape();
    uint64_t prod = std::accumulate(inp.begin(), inp.end(), 1, std::multiplies<int64_t>());

    int64_t* out = new int64_t[prod];
    __sigmoidFixedLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)out, 
        prod, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(inp, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* softmax_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    int n_tensor = pack.tensors.size();
    
    if (n_tensor != 1)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& inp = pack.tensors[0].shape();
    uint64_t prod = std::accumulate(inp.begin(), inp.end(), 1, std::multiplies<int64_t>());

    int64_t* out = new int64_t[prod];
    __softmaxFixedLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)out, 
        prod, 
        _error
    );

    if (*_error)
    {
        delete[] out;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(inp, out), 
        length_out
    );

    delete[] out;
    return out_bytes;
}

uint8_t* logsoftmax_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* softmax2d_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* reduction_max_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* reduction_min_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* reduction_mean_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* reduction_sum_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* reduction_argmax_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* reduction_argmin_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    *_error = true;
    return nullptr;
}

uint8_t* dropout_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 1)
    {
        *_error = true;
        return nullptr;
    }

    return abi_encode_tensor(pack.tensors[0], length_out);
}

uint8_t* globalavgpooling_call(const operation_pack& pack, int32_t* length_out, uint8_t* _error)
{
    if (pack.tensors.size() != 1 || pack.tensors[0].shape().size() < 3)
    {
        *_error = true;
        return nullptr;
    }

    const std::vector<uint64_t>& inp = pack.tensors[0].shape();
    std::vector<uint64_t> out_shape;
    
    for (int i = 0; i < inp.size() - 3; ++i)
    {
        out_shape.push_back(inp[i]);
    }

    out_shape.push_back(inp.back());
    
    int64_t prod = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>()); 
    int64_t* buffer = new int64_t[prod];

    __globalAvgPoolingFixedLongLong(
        (long long*)pack.tensors[0].data(), 
        (long long*)buffer, 
        prod / out_shape.back(), 1,
        out_shape.back(), 
        _error
    );

    if (*_error)
    {
        delete[] buffer;
        return nullptr;
    }

    uint8_t* out_bytes = abi_encode_tensor(
        TensorWrapper(out_shape, buffer), 
        length_out
    );

    delete[] buffer;
    return out_bytes;
}


int64_t read_opcode(const int64_t* data, uint8_t *__error)
{
    return data[3];
}

std::vector<int64_t> read_params(const int64_t* data, uint8_t *__error)
{
    const int64_t* data3 = data + 3;
    int64_t params_offset = data3[4] >> 3;
    int32_t n_params = data3[params_offset];
    std::vector<int64_t> params(n_params, 0);

    for (int i = 0; i < n_params; ++i)
    {
        params[i] = data3[params_offset + (i + 1) * 4];
    }

    return params;
}

std::vector<std::vector<uint64_t >> read_shapes(const int64_t* data, uint8_t *__error)
{
    const int64_t* data3 = data + 3;
    int64_t shapes_offset = data3[8] >> 3;
    int32_t n_tensor = data3[shapes_offset]; shapes_offset += 4;
    std::vector<std::vector<uint64_t>> shapes(n_tensor, std::vector<uint64_t>());

    for (int i = 0; i < n_tensor; ++i)
    {
        int offset = shapes_offset + (data3[shapes_offset + (i << 2)] >> 3);
        int ndims = data3[offset]; offset += 4; // count
        shapes[i].resize(ndims);

        for (int j = 0; j < ndims; ++j, offset += 4)
        {
            if (data3[offset] <= 0)
            {
                *__error = true;
                return {};
            }

            shapes[i][j] = data3[offset];
        }
    }

    return shapes;
}

std::vector<TensorWrapper> read_tensors(const int64_t* data, uint8_t *__error)
{
    const int64_t* data3 = data + 3;
    int64_t tensors_offset = data3[12] >> 3;
    std::vector<std::vector<uint64_t>> shapes = read_shapes(data, __error);

    if (*__error || data3[tensors_offset] != shapes.size())
    {
        *__error = true;
        std::cerr << "Error in reading shapes" << *__error << " " <<  data3[tensors_offset] << " " << shapes.size() << std::endl;
        return {};
    }

    tensors_offset += 4;
    std::vector<TensorWrapper> tensors;

    for (int i = 0; i < shapes.size(); ++i)
    {
        int offset = tensors_offset + (data3[tensors_offset + (i << 2)] >> 3);
        int cnt = data3[offset]; offset += 4;

        int prod = 1;
        for (const auto& x: shapes[i])
        {
            prod *= x;
        }
        
        if (((prod + 3) >> 2) != cnt)
        {
            *__error = true;
            return {};
        }

        cnt <<= 2;
        tensors.push_back(TensorWrapper(shapes[i], data + offset));
    }

    return tensors;
}

operation_pack abi_decode_op(const int64_t* inp, uint8_t *__error)
{
    auto opcode = read_opcode(inp, __error);
    auto params = read_params(inp, __error);
    auto tensors = read_tensors(inp, __error);

    operation_pack pack = {opcode, params, tensors};
    return pack;
}

uint8_t* abi_encode_tensor(const TensorWrapper& tensor, int32_t* length)
{
    const std::vector<uint64_t>& shape = tensor.shape();
    const int64_t* data = tensor.data();

    int64_t prod = 1;
    for (const int& x: shape)
    {
        prod *= x;
    }

    int64_t padded_prod = ((prod + 3) >> 2) << 2;

    // (2 offsets, 2 counts and a list of number) * 32 bytes + data (8 bytes) * prod
    *length = ((4 + shape.size()) << 5) + (padded_prod << 3); // bytes
    int64_t short_length = *length >> 3;
    int64_t* out_int64 = new int64_t[short_length];

    std::memset(out_int64, 0, short_length << 3);
    std::memcpy(out_int64 + 12, data, prod << 3);

    out_int64[3] = 64;
    out_int64[7] = (padded_prod + 12) << 3;
    out_int64[11] = padded_prod >> 2;

    out_int64[(out_int64[7] >> 3) + 3] = shape.size();
    for (int i = 0, offset = (out_int64[7] >> 3) + 7; i < shape.size(); ++i, offset += 4)
    {
        out_int64[offset] = shape[i];
    }

    uint8_t* out = new uint8_t[*length];

    for (int i = 0, j = 0; i < short_length; ++i, j += 8)
    {
        out[j + 0] = out_int64[i] >> 56;
        out[j + 1] = out_int64[i] >> 48;
        out[j + 2] = out_int64[i] >> 40;
        out[j + 3] = out_int64[i] >> 32;
        out[j + 4] = out_int64[i] >> 24;
        out[j + 5] = out_int64[i] >> 16;
        out[j + 6] = out_int64[i] >> 8;
        out[j + 7] = out_int64[i];
    }

    delete[] out_int64;
    return out;
}


// extern "C"
const uint8_t* cuda_execute_operation(
    uint8_t* payload_in, // bytes: opcode, params, shapes, tensors
    int32_t length_in, 
    int32_t* length_out,
    uint8_t* _error
)
{
    int short_length = length_in >> 3;
    int64_t* inp = new int64_t[short_length];
    memset(inp, 0, short_length);

    for (int i = 0, j = 0; i < short_length; ++i, j += 8)
    {
        inp[i] = (int64_t(payload_in[j + 0]) << 56) 
            | (int64_t(payload_in[j + 1]) << 48) 
            | (int64_t(payload_in[j + 2]) << 40) 
            | (int64_t(payload_in[j + 3]) << 32) 
            | (int64_t(payload_in[j + 4]) << 24) 
            | (int64_t(payload_in[j + 5]) << 16) 
            | (int64_t(payload_in[j + 6]) << 8) 
            |  int64_t(payload_in[j + 7]);
    }

    operation_pack pack = abi_decode_op(inp, _error);

    auto wrap_return_fn = [&](uint8_t* out = nullptr) -> uint8_t* {
        delete[] inp;

        if (out == nullptr)
        {
            *_error = true;
        }

        return out;
    };

    if (*_error)
    {
        return wrap_return_fn();
    }
    
    if (pack.op == opcode::MATMUL)
    {
        return wrap_return_fn(matmul_call(pack, length_out, _error));
    }

    if (pack.op == opcode::CONV2D)
    {
        return wrap_return_fn(conv2d_call(pack, length_out, _error));
    }

    if (pack.op == opcode::MAXPOOLING2D)
    {
        return wrap_return_fn(maxpooling2d_call(pack, length_out, _error));
    }

    if (pack.op == opcode::AVGPOOLING2D)
    {
        return wrap_return_fn(avgpooling2d_call(pack, length_out, _error));
    }

    if (pack.op == opcode::ELEMENTWISE_ADD)
    {
        return wrap_return_fn(elementwise_add_call(pack, length_out, _error));
    }

    if (pack.op == opcode::ELEMENTWISE_MUL)
    {
        return wrap_return_fn(elementwise_mul_call(pack, length_out, _error));
    }

    if (pack.op == opcode::ELEMENTWISE_SUB)
    {
        return wrap_return_fn(elementwise_sub_call(pack, length_out, _error));
    }

    if (pack.op == opcode::ELEMENTWISE_DIV)
    {
        return wrap_return_fn(elementwise_div_call(pack, length_out, _error));
    }

    if (pack.op == opcode::TRANSFORM_EXP)
    {
        return wrap_return_fn(transform_exp_call(pack, length_out, _error));
    }

    if (pack.op == opcode::TRANSFORM_SQRT)
    {
        return wrap_return_fn(transform_sqrt_call(pack, length_out, _error));
    }

    if (pack.op == opcode::BATCH_NORM)
    {
        return wrap_return_fn(batch_norm_call(pack, length_out, _error));
    }

    if (pack.op == opcode::LAYER_NORM)
    {
        return wrap_return_fn(layer_norm_call(pack, length_out, _error));
    }

    if (pack.op == opcode::ZSCORE)
    {
        return wrap_return_fn(zscore_call(pack, length_out, _error));
    }

    if (pack.op == opcode::MIN_MAX_SCALE)
    {
        return wrap_return_fn(min_max_scale_call(pack, length_out, _error));
    }

    if (pack.op == opcode::CONCATENATE)
    {
        return wrap_return_fn(concatenate_call(pack, length_out, _error));
    }

    if (pack.op == opcode::RELU)
    {
        return wrap_return_fn(relu_call(pack, length_out, _error));
    }

    if (pack.op == opcode::TANH)
    {
        return wrap_return_fn(tanh_call(pack, length_out, _error));
    }

    if (pack.op == opcode::SIGMOID)
    {
        return wrap_return_fn(sigmoid_call(pack, length_out, _error));
    }

    if (pack.op == opcode::SOFTMAX)
    {
        return wrap_return_fn(softmax_call(pack, length_out, _error));
    }

    if (pack.op == opcode::LOGSOFTMAX)
    {
        return wrap_return_fn(logsoftmax_call(pack, length_out, _error));
    }

    if (pack.op == opcode::SOFTMAX2D)
    {
        return wrap_return_fn(softmax2d_call(pack, length_out, _error));
    }

    if (pack.op == opcode::REDUCTION_MAX)
    {
        return wrap_return_fn(reduction_max_call(pack, length_out, _error));
    }

    if (pack.op == opcode::REDUCTION_MIN)
    {
        return wrap_return_fn(reduction_min_call(pack, length_out, _error));
    }

    if (pack.op == opcode::REDUCTION_MEAN)
    {
        return wrap_return_fn(reduction_mean_call(pack, length_out, _error));
    }

    if (pack.op == opcode::REDUCTION_SUM)
    {
        return wrap_return_fn(reduction_sum_call(pack, length_out, _error));
    }

    if (pack.op == opcode::REDUCTION_ARGMAX)
    {
        return wrap_return_fn(reduction_argmax_call(pack, length_out, _error));
    }

    if (pack.op == opcode::REDUCTION_ARGMIN)
    {
        return wrap_return_fn(reduction_argmin_call(pack, length_out, _error));
    }

    if (pack.op == opcode::DROPOUT)
    {
        return wrap_return_fn(dropout_call(pack, length_out, _error));
    }

    if (pack.op == opcode::GLOBAL_AVGPOOLING2D)
    {
        return wrap_return_fn(globalavgpooling_call(pack, length_out, _error));
    }

    return wrap_return_fn();
}

// extern "C"
void deallocate_cpp_response(const uint8_t* payload)
{
    if (payload != nullptr)
    {
        delete[] payload;
    }
}

// int main(int argc, char** argv)
// {
//     std::string file = argv[1];
    
//     std::ifstream inf(file, std::ios::binary);
//     inf.seekg(0, std::ios::end);
//     int nbytes = inf.tellg();
//     inf.seekg(0, std::ios::beg);

//     uint8_t* payload = new uint8_t[nbytes];

//     inf.read((char *) payload, nbytes);
    
//     bool _error = false;
//     int32_t length = 0;
//     uint8_t* output = cuda_execute_operation(payload, nbytes, &length, &_error);

//     delete[] payload;
//     delete[] output;
//     return 0;
// }
