import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from utils  import absolute_or_relative_error
from op import Operation, execute

def run_case(*args):
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    t1._data += 0.5

    exp_t1 = np.exp(t1.data)
    sqrt_t1 = np.sqrt(t1.data)
    
    exp_out = execute(Operation.TRANSFORM_EXP, [], t1)
    sqrt_out = execute(Operation.TRANSFORM_SQRT, [], t1)
    min_max_scale_out = execute(Operation.MIN_MAX_SCALE, [], t1)
    zscore_out = execute(Operation.ZSCORE, [], t1)

    _min, _max, _mean, _std = t1.data.min(), t1.data.max(), t1.data.mean(), t1.data.std()
    
    minmax_scale_t1 = (t1.data - _min) / (_max - _min)
    zscore_t1 = (t1.data - _mean) / _std
    
    diff_exp = absolute_or_relative_error(exp_out.data, exp_t1.flatten()).mean()
    diff_sqrt = absolute_or_relative_error(sqrt_out.data, sqrt_t1.flatten()).mean()
    diff_minmax_scale = absolute_or_relative_error(min_max_scale_out.data, minmax_scale_t1.flatten()).mean()
    diff_zscore = absolute_or_relative_error(zscore_out.data, zscore_t1.flatten()).mean()
    
    res = all([
        diff_exp < accepted_error,
        diff_sqrt < accepted_error,
        diff_minmax_scale < accepted_error,
        diff_zscore < accepted_error
    ])
    
    if not res:
        print('exp', diff_exp)
        print('sqrt', diff_sqrt)
        print('minmax_scale', diff_minmax_scale)
        print('zscore', diff_zscore)
    
    return res

def benchmark_element_wise():
    n_cases = 10000

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