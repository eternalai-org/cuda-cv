import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
from tensor import Tensor
from op import execute
from utils import compare_tensors

def run_case(*args):
    sample = (27, [], Tensor.random_tensor())
    tout = execute(*sample)

    if tout is None or not (compare_tensors(sample[2].data, tout.data) and compare_tensors(sample[2].shape, tout.shape)):
        random_payload = str(random.randint(0, 1000000))

        with open(f'error_{random_payload}.json', 'w') as f:
            json.dump({'sample': sample}, f)
            
        return False
    
    return True

def benchmark_abi():
    n_cases = 1

    futures = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for _ in tqdm(range(n_cases), total=n_cases, desc='Running test cases'):
            futures.append(executor.submit(run_case))

    fails = sum([not f.result() for f in futures])
    success = n_cases - fails

    print(f'Success: {success}/{n_cases}')
    print(f'Fails: {fails}/{n_cases}')

    if fails > 0:
        raise ValueError('Some test cases failed')

if __name__ == '__main__':
    benchmark_abi()