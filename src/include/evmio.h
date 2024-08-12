#pragma once
#include <stdint.h>

enum opcode
{
    CONV2D = 0, // 0
    MAXPOOLING2D = 1, // 1
    AVGPOOLING2D = 2, // 2

    // Matrix operations
    MATMUL = 3, // 3

    // Elementwise operations
    ELEMENTWISE_ADD = 4, // 4
    ELEMENTWISE_MUL = 5, // 5
    ELEMENTWISE_SUB = 6, // 6
    ELEMENTWISE_DIV = 7, // 7

    // Transforms
    TRANSFORM_EXP = 8, // 8
    TRANSFORM_SQRT = 9, // 9

    // Normalizations
    BATCH_NORM = 10, // 10
    LAYER_NORM = 11, // 11 
    ZSCORE = 12, // 12
    MIN_MAX_SCALE = 13, // 13

    // merging operations
    CONCATENATE = 14, // 14

    // Activations
    RELU = 15, // 15
    TANH = 16, // 16
    SIGMOID = 17, // 17
    SOFTMAX = 18, // 18
    LOGSOFTMAX = 19, // 19
    SOFTMAX2D = 20, // 20

    // Reductions
    REDUCTION_MAX = 21, // 21
    REDUCTION_MIN = 22, // 22
    REDUCTION_MEAN = 23, // 23
    REDUCTION_SUM = 24, // 24
    REDUCTION_ARGMAX = 25, // 25
    REDUCTION_ARGMIN = 26 // 26
};

#if __cplusplus
extern "C" {
#endif

void cuda_execute_operation(
    uint8_t* payload_in, 
    int32_t length_in,
    uint8_t* payload_out, 
    int32_t* length_out 
);

#if __cplusplus
}
#endif