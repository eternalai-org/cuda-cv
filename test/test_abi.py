import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
from tensor import Tensor
from op import execute

'''
uint8_t* cuda_execute_operation(
    uint8_t* payload_in, // bytes: opcode, params, shapes, tensors
    int32_t length_in, 
    int32_t* length_out,
    int8_t* has_error
);
'''

def create_random_tensor():
    shapes = [random.randint(1, 10) for _ in range(4)]
    flatten = np.prod(shapes)
    tensor = (np.random.rand(flatten).astype(np.float64) * 100).astype(np.int64)
    return Tensor(tensor, shapes)

def create_random_test_case():
    tensor = create_random_tensor()
    opcode, random_params = 27, [random.randint(0, 100) for _ in range(random.randint(0, 10))]
    return opcode, random_params, tensor    

def compare_tensors(t1, t2):
    return all(x == y for x, y in zip(t1, t2))


def run_case(*args):
    sample = create_random_test_case()
    tout = execute(*sample)

    if tout is None or not (compare_tensors(sample[2].data, tout.data) and compare_tensors(sample[2].shape, tout.shape)):
        random_payload = str(random.randint(0, 1000000))

        with open(f'error_{random_payload}.json', 'w') as f:
            json.dump({'sample': sample}, f)
            
        return False
    
    return True

def benchmark_abi():
    n_cases = 10 * 1000
    
    futures = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for _ in tqdm(range(n_cases), total=n_cases, desc='Running test cases'):
            futures.append(executor.submit(run_case))

    results = [f.result() for f in tqdm(futures, total=n_cases, desc='Processing results')]
    print('All results are', all(f == True for f in results))
    executor.shutdown()

if __name__ == '__main__':
    benchmark_abi()