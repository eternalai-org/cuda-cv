import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from op import Operation, execute
import random
import tensorflow as tf
from utils import absolute_or_relative_error, to_i64
import os

def run_case(*args): 
    spatial_size = random.randint(8, 256)
    channel_in = random.randint(1, 256)

    t1 = Tensor.random_tensor([spatial_size, spatial_size, channel_in])
    t1_data = t1.data.reshape(t1.shape)

    ma, mv, beta, gama = Tensor.random_tensor([channel_in]), Tensor.random_tensor([channel_in]), \
        Tensor.random_tensor([channel_in]), Tensor.random_tensor([channel_in])

    batch_norm_layer = tf.keras.layers.BatchNormalization()
    batch_norm_layer.build(t1_data.shape)
    batch_norm_layer.set_weights([gama.data, beta.data, ma.data, mv.data])
    
    eps = to_i64(int(1e-3 * 2 ** 32))

    expected = batch_norm_layer(t1_data).numpy().flatten()
    actual = execute(Operation.BATCH_NORM, [eps], [t1, ma, mv, gama, beta])

    err = absolute_or_relative_error(actual.data, expected).mean()
    res = err < 1e-4
    
    if not res:
        print(f'Error: {err}')

    return res

def benchmark_normaization():
    n_cases = 100

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
    benchmark_normaization()