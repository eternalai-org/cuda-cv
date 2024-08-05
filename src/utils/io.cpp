#include <operations.cuh>
#include <io.h>
#include <tensor.h>
#include <memory.h>

extern "C" void cuda_execute_operation(uint8_t* payload_in, uint8_t** payload_out, int& nBytes)
{
    int64_t* in = (int64_t*)payload_in;
}

extern "C" void deallocate(uint8_t* payload)
{
    delete[] payload;
}

// application
/*
payload := C.longlong([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
payload_out := None
nBytes := 0

C.cuda_execute_operation(payload, payload_out, nBytes)

// do something
C.deallocate(payload_out)
*/