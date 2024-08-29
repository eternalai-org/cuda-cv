import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from utils  import absolute_or_relative_error
from op import Operation, execute

def run_case(*args):
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()

    exp_t1 = np.exp(t1.data)
    sqrt_t1 = np.sqrt(t1.data)
    
    exp_out = execute(Operation.TRANSFORM_EXP, [], t1)
    sqrt_out = execute(Operation.TRANSFORM_SQRT, [], t1)
    
    diff_exp = absolute_or_relative_error(exp_out.data, exp_t1.flatten()).mean()
    diff_sqrt = absolute_or_relative_error(sqrt_out.data, sqrt_t1.flatten()).mean()
    
    return all([
        diff_exp < accepted_error,
        diff_sqrt < accepted_error
    ])

def benchmark_element_wise():
    n_cases = 1 * 1000

    futures = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for _ in tqdm(range(n_cases), total=n_cases, desc='Running test cases'):
            futures.append(executor.submit(run_case))

    fails = sum([not f.result() for f in futures])
    success = n_cases - fails

    print(f'Success: {success}/{n_cases}')
    print(f'Fails: {fails}/{n_cases}')

    if fails > 0:
        raise ValueError('Some test cases failed')

if __name__ == '__main__':
    benchmark_element_wise()