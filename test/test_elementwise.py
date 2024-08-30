import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from utils  import absolute_or_relative_error
from op import Operation, execute

def run_case(*args):
    accepted_error = 1e-4
    
    t1 = Tensor.random_tensor([1, 1])
    t2 = Tensor.random_tensor(t1.shape)
    
    expected_add = t1.data + t2.data
    expected_mul = t1.data * t2.data
    expected_sub = t1.data - t2.data
    expected_div = t1.data / t2.data
    
    add_out = execute(Operation.ELEMENTWISE_ADD, [], [t1, t2])
    mul_out = execute(Operation.ELEMENTWISE_MUL, [], [t1, t2])
    sub_out = execute(Operation.ELEMENTWISE_SUB, [], [t1, t2])
    div_out = execute(Operation.ELEMENTWISE_DIV, [], [t1, t2])
    
    mae_add = absolute_or_relative_error(add_out.data, expected_add)
    mae_mul = absolute_or_relative_error(mul_out.data, expected_mul)
    mae_sub = absolute_or_relative_error(sub_out.data, expected_sub)
    mae_div = absolute_or_relative_error(div_out.data, expected_div)
    
    x = all([
        mae_add < accepted_error,
        mae_mul < accepted_error,
        mae_sub < accepted_error,
        mae_div < accepted_error
    ])
    
    if not x:
        print(t1, t2)
        print(f'MAE ADD: {mae_add}')
        print(f'MAE MUL: {mae_mul}')
        print(f'MAE SUB: {mae_sub}')
        print(f'MAE DIV: {mae_div}')
        
    return x

def benchmark_element_wise():
    n_cases = 1000

    futures = []
    
    try:
        for _ in tqdm(range(n_cases), total=n_cases, desc='Running test cases'):
            futures.append(run_case())
    except KeyboardInterrupt:
        print('Interrupted')

    fails = sum([not f for f in futures])
    success = len(futures) - fails

    print(f'Success: {success}/{len(futures)}')
    print(f'Fails: {fails}/{len(futures)}')

    if fails > 0:
        raise ValueError('Some test cases failed')

if __name__ == '__main__':
    benchmark_element_wise()