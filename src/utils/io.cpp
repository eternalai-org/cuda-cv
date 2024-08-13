#include <operations.cuh>
#include <evmio.h>
#include <tensor.h>
#include <memory.h>
#include <iostream>
#include <fstream> 
#include <bitset>
#include <cstring>
#include <vector>
#include <algorithm>
#include <helpers.cuh>

int64_t read_opcode(const int64_t* data, bool& _error)
{
    return data[3];
}

std::vector<int64_t> read_params(const int64_t* data, bool& _error)
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

std::vector<std::vector<uint32_t >> read_shapes(const int64_t* data, bool& _error)
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
                _error = true;
                return {};
            }

            shapes[i][j] = data3[offset];
        }
    }

    return shapes;
}

std::vector<TensorWrapper> read_tensors(const int64_t* data, bool& _error)
{
    const int64_t* data3 = data + 3;
    int64_t tensors_offset = data3[12] >> 3;
    std::vector<std::vector<uint32_t>> shapes = read_shapes(data, _error);

    if (_error || data3[tensors_offset] != shapes.size())
    {
        _error = true;
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
            _error = true;
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

operation_pack abi_decode(const int64_t* inp, bool& _error)
{
    operation_pack pack = {
        read_opcode(inp, _error),
        read_params(inp, _error),
        read_tensors(inp, _error)
    };

    return pack;
}

int8_t* abi_encode_tensor(const TensorWrapper& tensor, int32_t& length)
{
    const std::vector<uint32_t>& shape = tensor.shape();
    const int64_t* data = tensor.data();

    int32_t prod = 1;
    for (const int& x: shape)
    {
        prod *= x;
    }

    length = (4 + shape.size() + prod + 4 - (prod % 4)) << 5;
    int short_length = (4 + shape.size() + prod + 4 - (prod % 4)) << 2;
    int64_t* out_int64 = new int64_t[short_length];

    std::memset(out_int64, 0, short_length << 3);

    out_int64[3] = 64;
    out_int64[7] = ((prod + 3) << 3) + 96;
    out_int64[11] = (prod + 3) / 4;
    for (int i = 0; i < prod; ++i)
    {
        out_int64[12 + i] = data[i];
    }

    out_int64[out_int64[7] >> 3] = shape.size();
    for (int i = 0, offset = (out_int64[7] >> 3) + 4; i < shape.size(); ++i, offset += 4)
    {
        out_int64[offset + 3] = shape[i];
    }

    int8_t* out = new int8_t[length];

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

void cuda_execute_operation(
    uint8_t* payload_in, // bytes: opcode, params, shapes, tensors
    int32_t length_in, 
    uint8_t** payload_out, // bytes abi encode
    int32_t* length_out,
    bool& has_error
)
{
    int64_t* inp = new int64_t[length_in / sizeof(int64_t)];

    memset(inp, 0, length_in / sizeof(int64_t));

    for (int i = 0, j = 0; i < (length_in >> 3); ++i, j += 8)
    {
        inp[i] = (int64_t(payload_in[j + 0]) << 56) 
            | (int64_t(payload_in[j + 1]) << 48) 
            | (int64_t(payload_in[j + 2]) << 40) 
            | (int64_t(payload_in[j + 3]) << 32) 
            | (int64_t(payload_in[j + 4]) << 24) 
            | (int64_t(payload_in[j + 5]) << 16) 
            | (int64_t(payload_in[j + 6]) << 8) 
            | int64_t(payload_in[j + 7]);
    }

    operation_pack pack = abi_decode(inp, has_error);
    
    if (has_error)
    {
        std::cout << "Error in decoding" << std::endl;
        delete[] inp;
        return;
    }

    std::cout << "Operation: " << pack.op << std::endl;
    std::cout << "Params: " << pack.params << std::endl;
    std::cout << "Tensors: " << std::endl;
    for (const auto& tensor: pack.tensors)
    {
        std::cout << tensor << std::endl;
    }

    delete[] inp;
}

void deallocate(uint8_t* payload)
{
    delete[] payload;
}

int main(int argc, char** argv)
{
    std::string file = argv[1];
    
    std::ifstream inf(file, std::ios::binary);
    inf.seekg(0, std::ios::end);
    int nbytes = inf.tellg();
    inf.seekg(0, std::ios::beg);

    uint8_t* payload = new uint8_t[nbytes];

    inf.read((char *) payload, nbytes);
    
    bool has_error = false;
    cuda_execute_operation(payload, nbytes, nullptr, nullptr, has_error);

    delete[] payload;


    std::vector<uint32_t> shapes = {2, 3, 4};
    int prod = 1;

    for (const auto& x: shapes)
    {
        prod *= x;
    }

    int64_t* data = new int64_t[prod + 4 - (prod % 4)];

    for (int i = 0; i < prod; ++i)
    {
        data[i] = i;
    }

    TensorWrapper tensor(shapes, data);

    int32_t length = 0;
    int8_t* bytes = abi_encode_tensor(tensor, length);

    for (int i = 0; i < length; ++i)
    {
        std::cout << int(bytes[i]) << " ";
    }

    delete[] data;
    delete[] bytes;
    return 0;
}
