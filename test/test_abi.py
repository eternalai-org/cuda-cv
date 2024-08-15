from eth_abi import encode as abi_encode, decode as abi_decode
import ctypes
import os
import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

'''
uint8_t* cuda_execute_operation(
    uint8_t* payload_in, // bytes: opcode, params, shapes, tensors
    int32_t length_in, 
    int32_t* length_out,
    int8_t* has_error
);
'''

dll = ctypes.CDLL(os.path.join(os.getcwd(), 'libcomputelib.so'))

def compress_uint256(a1 = 0, a2 = 0, a3 = 0, a4 = 0):
        return a4 + (a3 << 64) + (a2 << 128) + (a1 << 192)
    
def unpack_uint256(a):
    a1 = (a >> 192) & 0xFFFFFFFFFFFFFFFF
    a2 = (a >> 128) & 0xFFFFFFFFFFFFFFFF
    a3 = (a >> 64) & 0xFFFFFFFFFFFFFFFF
    a4 = a & 0xFFFFFFFFFFFFFFFF
    
    return a1, a2, a3, a4

def chunked(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def create_random_tensor():
    shapes = [random.randint(1, 10) for _ in range(4)]
    flatten = np.prod(shapes)

    tensor = (np.random.rand(flatten).astype(np.float64) * 100).astype(np.int64).tolist()
    tensor = [compress_uint256(*i) for i in chunked(tensor, 4)]
    
    return tensor, shapes

def create_random_test_case():
    tensor, shapes = create_random_tensor()
    
    opcode, random_params = 27, [random.randint(0, 100) for _ in range(random.randint(0, 10))]
    return opcode, random_params, [shapes], [tensor]    

def compare_tensors(t1, t2):
    return all(x == y for x, y in zip(t1, t2))

def benchmark_abi():
    

    import time, random

    n_cases = 1 * 1000 * 1000
    total_execution_time = 0
    
    def run_case(*args):
        sample = create_random_test_case()
        inp = abi_encode(('uint64', 'uint64[]', 'uint64[][]', 'uint256[][]'), sample)
        
        length_out = ctypes.c_int()
        has_error = ctypes.c_int()

        length_out_ptr = ctypes.pointer(length_out)
        has_error_ptr = ctypes.pointer(has_error)

        out = dll.cuda_execute_operation(inp, len(inp), length_out_ptr, has_error_ptr)

        deref = bytes(ctypes.cast(out, ctypes.POINTER(ctypes.c_ubyte * length_out.value)).contents)

        template_out = ('uint256[]', 'uint64[]')

        (tensor_out, shape_out) = abi_decode(template_out, deref)
        dll.deallocate(out)

        if not (compare_tensors(sample[3][0], tensor_out) and compare_tensors(sample[2][0], shape_out) and has_error.value == 0):
            random_payload = str(random.randint(0, 1000000))

            with open(f'error_{random_payload}.bin', 'wb') as f:
                f.write(deref)
                
        return compare_tensors(sample[3][0], tensor_out) and compare_tensors(sample[2][0], shape_out) and has_error.value == 0

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(run_case, range(n_cases)), total=n_cases))

    print('All results are', all(results))

if __name__ == '__main__':
    # test matmul
    shape_a = [2, 3]
    shape_b = [3, 2]
    a = (np.random.rand(*shape_a).astype(np.float64) * 100).astype(np.int64)
    b = (np.random.rand(*shape_b).astype(np.float64) * 100).astype(np.int64)
    c = np.matmul(a, b)

    print('Expected:', c)

    opcode = 3 
    params = []
    shapes = [shape_a, shape_b]
    _a = [compress_uint256(*i) for i in chunked((a.flatten() * (2 ** 32)).tolist(), 4)]
    _b = [compress_uint256(*i) for i in chunked((b.flatten() * (2 ** 32)).tolist(), 4)]
    tensors = [_a, _b]
    
    sample = opcode, params, shapes, tensors
    inp = abi_encode(('uint64', 'uint64[]', 'uint64[][]', 'uint256[][]'), sample)
    
    length_out = ctypes.c_int()
    has_error = ctypes.c_int()
    
    length_out_ptr = ctypes.pointer(length_out)
    has_error_ptr = ctypes.pointer(has_error)
    
    out = dll.cuda_execute_operation(inp, len(inp), length_out_ptr, has_error_ptr) 
    
    deref = bytes(ctypes.cast(out, ctypes.POINTER(ctypes.c_ubyte * length_out.value)).contents)
    template_out = ('uint256[]', 'uint64[]')
    
    (tensor_out, shape_out) = abi_decode(template_out, deref)
    
    dll.deallocate_cpp_response(out)
    
    tensor_out = np.array([unpack_uint256(i) for i in tensor_out], dtype=np.float64).flatten().reshape(shape_out)
        
    print('Result:', c)
    print('Got:', tensor_out / (2 ** 32))