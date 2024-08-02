#pragma once
#include <stdint.h>

enum opcode
{
    CONV2D = 0,
    MAX_POOLING = 1,
    AVG_POOLING = 2,
    GLOBAL_AVG_POOLING = 3,

    MAT_ADD = 4,
    MAT_SUB = 5,
    MAT_MUL = 6,
    MAT_DIV = 7,
    MAT_SQRT = 8,

    SUM_REDUCTION = 9,
    AVG_REDUCTION = 10,
    MAX_REDUCTION = 11,
    MIN_REDUCTION = 12,
    MEAN_REDUCTION = 13,
    STD_REDUCTION = 14,
    MAX_MIN_SCALE = 15,
    Z_SCORE = 16,

    SOFTMAX = 17,
    SIGMOID = 18,
    TANH = 19,
    RELU = 20,

    LAYER_NORMALIZE = 25,
    BATCH_NORMALIZE = 26,
    MAXMUL = 27,
    MAXMUL_FLOAT = 28,
    MAXMUL_LONG = 29,
    MAXMUL_INT = 30,
    MAXMUL_DOUBLE = 31,
    CONCATENATE = 32
};

#if __cplusplus
extern "C" {
#endif

void cuda_execute_operation(uint8_t* payload_in, uint8_t* payload_out);

#if __cplusplus
}
#endif