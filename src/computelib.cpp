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

int64_t read_opcode(const int64_t* data, uint8_t *_eerror)
{
    return data[3];
}

std::vector<int64_t> read_params(const int64_t* data, uint8_t *_eerror)
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

std::vector<std::vector<uint32_t >> read_shapes(const int64_t* data, uint8_t *_eerror)
{
    const int64_t* data3 = data + 3;
    int64_t shapes_offset = data3[8] >> 3;
    int32_t n_tensor = data3[shapes_offset]; shapes_offset += 4;
    std::vector<std::vector<uint32_t>> shapes(n_tensor, std::vector<uint32_t>());

    for (int i = 0; i < n_tensor; ++i)
    {
        int offset = shapes_offset + (data3[shapes_offset + (i << 2)] >> 3);
        int ndims = data3[offset]; offset += 4; // count
        shapes[i].resize(ndims);

        for (int j = 0; j < ndims; ++j, offset += 4)
        {
            if (data3[offset] <= 0)
            {
                *_eerror = true;
                return {};
            }

            shapes[i][j] = data3[offset];
        }
    }

    return shapes;
}

std::vector<TensorWrapper> read_tensors(const int64_t* data, uint8_t *_eerror)
{
    const int64_t* data3 = data + 3;
    int64_t tensors_offset = data3[12] >> 3;
    std::vector<std::vector<uint32_t>> shapes = read_shapes(data, _eerror);

    if (*_eerror || data3[tensors_offset] != shapes.size())
    {
        *_eerror = true;
        std::cerr << "Error in reading shapes" << *_eerror << " " <<  data3[tensors_offset] << " " << shapes.size() << std::endl;
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
            *_eerror = true;
            return {};
        }

        cnt <<= 2;
        tensors.push_back(TensorWrapper(shapes[i], data + offset));
    }

    return tensors;
}

struct operation_pack
{
    int64_t op;
    std::vector<int64_t> params;
    std::vector<TensorWrapper> tensors;  
};

operation_pack abi_decode_op(const int64_t* inp, uint8_t *_eerror)
{
    auto opcode = read_opcode(inp, _eerror);
    auto params = read_params(inp, _eerror);
    auto tensors = read_tensors(inp, _eerror);

    operation_pack pack = {opcode, params, tensors};
    return pack;
}

uint8_t* abi_encode_tensor(const TensorWrapper& tensor, int32_t& length)
{
    const std::vector<uint32_t>& shape = tensor.shape();
    const int64_t* data = tensor.data();

    int32_t prod = 1;
    for (const int& x: shape)
    {
        prod *= x;
    }

    int padded_prod = ((prod + 3) >> 2) << 2;

    // (2 offsets, 2 counts and a list of number) * 32 bytes + data (8 bytes) * prod
    length = ((4 + shape.size()) << 5) + (padded_prod << 3); // bytes
    int short_length = length >> 3;
    int64_t* out_int64 = new int64_t[short_length];

    std::memset(out_int64, 0, short_length << 3);
    std::memcpy(out_int64 + 12, data, prod << 3);

    out_int64[3] = 64;
    out_int64[7] = (padded_prod + 12) << 3;
    out_int64[11] = padded_prod >> 2;

    out_int64[(out_int64[7] >> 3) + 3] = shape.size();
    for (
        int i = 0, offset = (out_int64[7] >> 3) + 4; 
        i < shape.size(); 
        ++i, offset += 4
    )
    {
        out_int64[offset + 3] = shape[i];
    }

    uint8_t* out = new uint8_t[length];

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

uint8_t* cuda_execute_operation(
    uint8_t* payload_in, // bytes: opcode, params, shapes, tensors
    int32_t length_in, 
    int32_t* length_out,
    uint8_t* has_eerror
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

    operation_pack pack = abi_decode_op(inp, has_eerror);

    auto wrap_return_fn = [&](uint8_t* out = nullptr) -> uint8_t* {
        delete[] inp;

        if (out == nullptr)
        {
            *has_eerror = true;
        }

        return out;
    };

    if (*has_eerror)
    {
        return wrap_return_fn();
    }

    if (pack.op == opcode::DROPOUT)
    {
        if (pack.tensors.size() == 0) {
            return wrap_return_fn();
        }

        uint8_t* out = abi_encode_tensor(pack.tensors[0], *length_out);
        return wrap_return_fn(out);
    }
    
    if (pack.op == opcode::MATMUL)
    {
        if (pack.tensors.size() != 2)
        {
            return wrap_return_fn();
        }

        uint32_t h1, w1, h2, w2;

        w1 = *pack.tensors[0].shape().rbegin();
        w2 = *pack.tensors[1].shape().rbegin();
        h1 = 1, h2 = 1;

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
            return wrap_return_fn();
        }

        std::vector<uint32_t> out_shape = {h1, w2};
        int64_t* out = new int64_t[h1 * w2];  

        __maxmulFixedLongLong(
            pack.tensors[0].data(), 
            pack.tensors[1].data(), 
            out, h1, w1, w2, 
            has_eerror
        );

        if (*has_eerror)
        {
            delete[] out;
            return wrap_return_fn();
        }

        uint8_t* out_bytes = abi_encode_tensor(
            TensorWrapper(out_shape, out), 
            *length_out
        );

        delete[] out;
        return wrap_return_fn(out_bytes);
    }

    return wrap_return_fn();
}

void deallocate(uint8_t* payload)
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
    
//     bool has_eerror = false;
//     int32_t length = 0;
//     uint8_t* output = cuda_execute_operation(payload, nbytes, &length, &has_eerror);

//     delete[] payload;
//     delete[] output;
//     return 0;
// }
