#ifndef __TENSOR_H__
#define __TENSOR_H__
#include <vector>
#include <stdint.h>
#include <cstring>
#include <assert.h>
#include <memory>

class  Tensor {
public:
    std::vector<uint16_t> mshape;
    uint64_t* mdata;

    Tensor() : mshape({}), mdata(nullptr) {}

    Tensor(const std::vector<uint16_t>& shape, uint64_t* data = nullptr) : mshape(shape) { 
        if (data != nullptr)
        {
            uint64_t prod = 1;

            for (const auto& dim : mshape) {
                prod *= dim;
            }

            mdata = new uint64_t[prod];
            memcpy(mdata, data, prod * sizeof(uint64_t));
        }
        else 
        {
            mdata = nullptr;
        }
    }

    Tensor(const Tensor& other) {
        mshape = other.mshape;

        uint64_t prod = 1;
        for (const auto& dim : mshape) {
            prod *= dim;
        }

        mdata = new uint64_t[prod];
        memcpy(mdata, other.mdata, prod * sizeof(uint64_t));
    }

    Tensor& assign(uint64_t* data)
    {
        if (this->mdata != nullptr)
        {
            delete[] this->mdata;
        }

        this->mdata = data;
        return *this;
    }

    ~Tensor() {
        if (mdata != nullptr) {
            delete[] mdata;
        }
    }

    // Tensor& unpack(uint8_t* payload, int size)
    static Tensor* unpack(uint8_t* payload)
    {
        uint64_t* ptr = (uint64_t*) payload;
        int dims = ptr[0];
        std::vector<uint16_t> shape(ptr + 1, ptr + 1 + dims);
                
        int64_t prod = 1;

        for (const auto& dim : shape) {
            prod *= dim;
        }

        uint64_t* data = new uint64_t[prod];
        memcpy(data, ptr + 1 + dims, prod * sizeof(uint64_t));

        return new Tensor(shape, data);
    }

    static Tensor* unpack(std::shared_ptr<uint8_t> payload)
    {
        return unpack(payload.get());
    }

    std::shared_ptr<uint8_t[]> pack()
    {
        uint64_t prod = 1;
        for (const auto& x : mshape)
        {
            prod *= x;
        }

        std::shared_ptr<uint8_t[]> payload(new uint8_t[(1 + this->dims() + prod) * sizeof(uint64_t)]);
        uint64_t* ptr = (uint64_t*) payload.get();
        
        ptr[0] = this->dims();
        for (int i = 0; i < this->dims(); ++i)
        {
            ptr[i + 1] = mshape[i];
        }

        memcpy(ptr + 1 + this->dims(), mdata, prod * sizeof(uint64_t));
        return payload;
    }

    int dims()
    {
        return mshape.size();
    }

    std::vector<uint16_t> shape()
    {
        return mshape;
    }
    
    Tensor& reshape(std::vector<uint16_t> shape)
    {
        uint32_t prod_1 = 1, prod_2 = 1;

        for (const auto& dim : mshape) {
            prod_1 *= dim;
        }

        for (const auto& dim : shape) {
            prod_2 *= dim;
        }

        assert(prod_1 == prod_2);

        mshape = shape;
        return *this;
    }
};


#endif