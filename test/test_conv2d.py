import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
from tensor import Tensor
from op import execute, Operation
import keras

def run_case(*args):
    eps = 1e-4

    spatial_size = random.randint(8, 10)
    channel_in = random.randint(1, 4)
    channel_out = random.randint(1, 4)
    
    t = Tensor.random_tensor([spatial_size, spatial_size, channel_in])
    
    ksize = random.randint(1, min(16, spatial_size))
    stride = random.randint(1, ksize)
    padding = random.choice(['valid', 'same'])
    padding_i = 1 if padding == 'same' else 0
    
    random_kernel = Tensor.random_tensor((ksize, ksize, channel_in, channel_out))
    random_bias = Tensor.random_tensor((channel_out,))

    print(random_kernel.shape)

    params = [stride, stride, padding_i]
    
    conv2d = keras.layers.Conv2D(
        filters=channel_out, 
        kernel_size=(ksize, ksize), 
        strides=(stride, stride), 
        padding=padding
    )
    
    conv2d.build(t.data.reshape(1, *t.shape).shape)
    conv2d.set_weights([random_kernel.data.reshape(random_kernel.shape), random_bias.data.reshape(random_bias.shape)])
    
    expected_conv2d = conv2d(t.data.reshape(1, *t.shape)).numpy().flatten()

    conv2d_out = execute(Operation.CONV2D, params, [t, random_kernel, random_bias])

    conv2d_mae = np.abs(conv2d_out.data - expected_conv2d).mean()
    
    res = conv2d_mae < eps
    
    if not res:
        print(padding)
        print(f'Conv2D MAE: {conv2d_mae}')

        print(f'Expected: {expected_conv2d}')
        print(f'Actual: {conv2d_out.data}')
        
    return res

def benchmark_conv2d():
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
    benchmark_conv2d()