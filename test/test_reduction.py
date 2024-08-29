import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from utils  import absolute_or_relative_error
from op import Operation, execute
import os

def manual_test():
    t = Tensor([1, 2, 3, 4, 5, 6], (1, 2, 3))
    channel_wise = execute(Operation.CHANNEL_WISE_SUM_REDUCTION, [2], t) # axis 2
    print(t.data.reshape(t.shape))
    print(channel_wise.data)
    
def run_case(*args):
    accepted_error = 1e-4
    
    t1 = Tensor.random_tensor()
    
    expected_mean_t1 = np.mean(t1.data)
    expected_max_t1 = np.max(t1.data)
    expected_min_t1 = np.min(t1.data)
    expected_sum_t1 = np.sum(t1.data)
    
    
    mean_out = execute(Operation.REDUCTION_MEAN, [], t1)
    max_out = execute(Operation.REDUCTION_MAX, [], t1)
    min_out = execute(Operation.REDUCTION_MIN, [], t1)
    sum_out = execute(Operation.REDUCTION_SUM, [], t1)
    
    
    diff_mean = absolute_or_relative_error(mean_out.data, expected_mean_t1)
    diff_max = absolute_or_relative_error(max_out.data, expected_max_t1)
    diff_min = absolute_or_relative_error(min_out.data, expected_min_t1)
    diff_sum = absolute_or_relative_error(sum_out.data, expected_sum_t1)
    
    print('mean', mean_out.data, expected_mean_t1)
    print('max', max_out.data, expected_max_t1)
    print('min', min_out.data, expected_min_t1)
    print('sum', sum_out.data, expected_sum_t1)
    
    return all([
        diff_mean < accepted_error,
        diff_max < accepted_error,
        diff_min < accepted_error,
        diff_sum < accepted_error
    ])

def benchmark_element_wise():
    n_cases = 1

    futures = []
    for _ in tqdm(range(n_cases), total=n_cases, desc='Running test cases'):
        futures.append(run_case())

    fails = sum([not f for f in futures])
    success = n_cases - fails

    print(f'Success: {success}/{n_cases}')
    print(f'Fails: {fails}/{n_cases}')

    if fails > 0:
        raise ValueError('Some test cases failed')

if __name__ == '__main__':
    # os.environ['BENCHMARK_LOGGING_SILENT'] = '1'
    # benchmark_element_wise()
    manual_test()