import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from op import Operation, execute
import random
from keras import layers
from utils import absolute_or_relative_error, to_i64
import os

def run_case(*args): 
    spatial_size = 32 # random.randint(8, 256)
    channel_in = 32 # random.randint(1, 16)

    t1 = Tensor.random_tensor([spatial_size, spatial_size, channel_in])
    t1_data = t1.data.reshape(t1.shape)

    ma, mv, beta, gama = Tensor.random_tensor([channel_in]), Tensor.random_tensor([channel_in]), \
        Tensor.random_tensor([channel_in]), Tensor.random_tensor([channel_in])
        
    mv._data += 0.5

    batch_norm_layer = layers.BatchNormalization()
    batch_norm_layer.build(t1_data.shape)
    batch_norm_layer.set_weights([gama.data, beta.data, ma.data, mv.data])
    
    eps = to_i64(int(1e-3 * 2 ** 32))

    expected = batch_norm_layer(t1_data).numpy().flatten()
    actual = execute(Operation.BATCH_NORM, [eps], [t1, gama, beta, ma, mv])

    err = absolute_or_relative_error(actual.data, expected).mean()
    res = err < 1e-4
    
    if not res:
        print(f'Error: {err}')

    return res

def benchmark_normaization():
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
    # os.environ['BENCHMARK_LOGGING_SILENT'] = '1'
    benchmark_normaization()