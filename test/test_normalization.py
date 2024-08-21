import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from op import Operation, execute
import random
import tensorflow as tf

def run_case(*args): 
    spatial_size = random.randint(16, 1024)
    channel_in = random.randint(1, 255)
     
    t1 = Tensor.random_tensor([spatial_size, spatial_size, channel_in])
    t1_data = t1.data.reshape(t1.shape)
    
    ma, mv, beta, gama = Tensor.random_tensor([channel_in]), Tensor.random_tensor([channel_in]), \
        Tensor.random_tensor([channel_in]), Tensor.random_tensor([channel_in])
    
    batch_norm_layer = tf.keras.layers.BatchNormalization()
    batch_norm_layer.build(t1_data.shape)
    batch_norm_layer.set_weights([ma.data, mv.data, beta.data, gama.data])
    

def benchmark_normaization():
    n_cases = 10 * 1000

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
    benchmark_normaization()