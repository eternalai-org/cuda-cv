#include <operations.cuh>
#include <evmio.h>
#include <tensor.h>
#include <memory.h>
#include <iostream>
#include <fstream> 
#include <bitset>
#include <cstring>
#include <vector>

int64_t read_opcode(const int64_t* data)
{
    return data[3];
}

std::vector<int64_t> read_params(const int64_t* data)
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

std::vector<std::vector<int>> read_shapes(const int64_t* data)
{
    const int64_t* data3 = data + 3;
    int64_t shapes_offset = data3[8] >> 3;
    int32_t n_tensor = data3[shapes_offset];
    std::vector<std::vector<int>> shapes(n_tensor, std::vector<int>());

    for (int i = 0; i < n_tensor; ++i)
    {
        int offset = data3[shapes_offset + (i + 1) * 4] >> 3;
        std::cout << "DEBUG 0: " << offset << '\n';
    
        int ndims = data3[offset];
        shapes[i].resize(ndims);
        std::cout << "DEBUG 1: " << ndims << '\n';

        for (int j = 0; j < ndims; ++j)
        {
            shapes[i][j] = data3[shapes_offset + (i + 1) * 4 + j + 1];
        }
    }

    return shapes;
}

std::vector<Tensor> read_tensors(const int64_t* data)
{
    const int64_t* data3 = data + 3;
    int64_t tensors_offset = data3[12];

    std::vector<std::vector<int>> shapes = read_shapes(data);
    std::vector<Tensor> tensors;

    return tensors;
}

void cuda_execute_operation(
    uint8_t* payload_in, // bytes: opcode, params, shapes, tensors
    int32_t length_in, 
    uint8_t* payload_out, // bytes abi encode
    int32_t* length_out
)
{
    int64_t* inp = new int64_t[length_in / sizeof(int64_t)];

    memset(inp, 0, length_in / sizeof(int64_t));

    for (int i = 0, x = 1; i < length_in; ++i)
    {
        inp[i / sizeof(int64_t)] += ((uint64_t) payload_in[i] << (64 - (1 + i) % sizeof(int64_t) * 8));
    }

    std::vector<int64_t> params = read_params(inp);
    std::vector<Tensor> tensors = read_tensors(inp);

    for (auto x: params)
    {
        std::cout << x << '\n';
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

    cuda_execute_operation(payload, nbytes, nullptr, nullptr);

    delete[] payload;

    return 0;
}
