import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
from tensor import Tensor
from op import execute, Operation
import tensorflow as tf
import time

def run_case(*args):
    eps = 1e-4

    spatial_size = 7 # random.randint(8, 256)
    channel_in = 992 # random.randint(1, 256)
    channel_out = 128 # random.randint(1, 256)

    t = Tensor.random_tensor([spatial_size, spatial_size, channel_in])

    ksize = 1 # random.randint(1, min(16, spatial_size))
    stride = 1 # random.randint(1, ksize)
    padding = random.choice(['valid', 'same'])
    padding_i = 1 if padding == 'same' else 0
    
    random_kernel = Tensor.random_tensor((ksize, ksize, channel_in, channel_out))
    random_bias = Tensor.random_tensor((channel_out,))

    params = [stride, stride, padding_i]
    
    conv2d = tf.keras.layers.Conv2D(
        filters=channel_out, 
        kernel_size=(ksize, ksize), 
        strides=(stride, stride), 
        padding=padding
    )
    
    conv2d.build(t.data.reshape(1, *t.shape).shape)
    conv2d.set_weights([random_kernel.data.reshape(random_kernel.shape), random_bias.data.reshape(random_bias.shape)])
    
    t_start = time.time()
    expected_conv2d = conv2d(t.data.reshape(1, *t.shape)).numpy().flatten()
    print('Tensorflow cpu', f'Elapsed time: {time.time() - t_start}')

    conv2d_out = execute(Operation.CONV2D, params, [t, random_kernel, random_bias])

    conv2d_mae = np.abs(conv2d_out.data - expected_conv2d).mean()
    
    res = conv2d_mae < eps
    
    if not res:
        print(padding)
        print(f'Conv2D MAE: {conv2d_mae}')

        print("kernel", random_kernel.shape)
        print("bias", random_bias.shape)

        print(f'Input shape: {t.shape}')
        print(f'Expected: {expected_conv2d.shape}')
        print(f'Actual: {conv2d_out.shape}')
        
    return res

def benchmark_conv2d():
    n_cases = 100

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
    benchmark_conv2d()