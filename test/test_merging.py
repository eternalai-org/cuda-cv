import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from op import Operation, execute
import traceback

def run_case(*args): 
    t1 = Tensor.random_tensor()
    t2 = Tensor.random_tensor(t1.shape)
    t3 = Tensor.random_tensor(t1.shape)
    
    t1_data, t2_data, t3_data = t1.data.reshape(t1.shape), t2.data.reshape(t2.shape), t3.data.reshape(t3.shape)
    axis = np.random.randint(0, len(t1.shape))
    
    expected = np.concatenate([t1_data, t2_data, t3_data], axis=axis)
    params = [axis]
    
    out = execute(Operation.CONCATENATE, params, [t1, t2, t3])
    return np.allclose(out.data.reshape(out.shape), expected)

def benchmark_concatenate():
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
    benchmark_concatenate()