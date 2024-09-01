import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import keras
import random
from tensor import Tensor
from op import Operation, execute
from utils import absolute_or_relative_error, log as wrap_log
from .test_registry import wrap_test
import time

@wrap_test(
    name='maxpooling2d test',
    repeat=1000,
    meta={
        'description': 'Test max pooling operation',
        'accepted_error': 1e-4
    }
)
def max_pooling_correct_check(*args):
    eps = 1e-4

    spatial_size = random.randint(16, 512)
    channel_in = random.randint(1, 512)
    
    t = Tensor.random_tensor([spatial_size, spatial_size, channel_in])
    window_size = random.randint(1, min(16, spatial_size))
    stride = random.randint(1, window_size)
    padding = random.choice(['valid', 'same'])

    padding_i = 1 if padding == 'same' else 0
    maxpooling = keras.layers.MaxPooling2D(
        pool_size=(window_size, window_size), 
        strides=(stride, stride), 
        padding=padding
    )
    
    params = [window_size, window_size, stride, stride, padding_i]
    
    t_start = time.time()
    expected_max_pooling = maxpooling(t.data.reshape(1, *t.shape)).numpy().flatten()
    wrap_log('Tensorflow cpu', f'Elapsed time: {time.time() - t_start}')

    maxpooling_out = execute(Operation.MAXPOOLING2D, params, t)
    maxpooling_mae = absolute_or_relative_error(maxpooling_out.data, expected_max_pooling).mean()
    
    return maxpooling_mae < eps


@wrap_test(
    name='avgpooling2d test',
    repeat=1000,
    meta={
        'description': 'Test average pooling operation',
        'accepted_error': 1e-4
    }
)
def avg_pooling_correct_check():
    eps = 1e-4

    spatial_size = random.randint(16, 512)
    channel_in = random.randint(1, 512)
    
    t = Tensor.random_tensor([spatial_size, spatial_size, channel_in])
    window_size = random.randint(1, min(16, spatial_size))
    stride = random.randint(1, window_size)
    padding = random.choice(['valid', 'same'])

    padding_i = 1 if padding == 'same' else 0
    maxpooling = keras.layers.AveragePooling2D(
        pool_size=(window_size, window_size), 
        strides=(stride, stride), 
        padding=padding
    )
    
    params = [window_size, window_size, stride, stride, padding_i]
    
    t_start = time.time()
    expected_max_pooling = maxpooling(t.data.reshape(1, *t.shape)).numpy().flatten()
    wrap_log('Tensorflow cpu', f'Elapsed time: {time.time() - t_start}')

    maxpooling_out = execute(Operation.AVGPOOLING2D, params, t)
    maxpooling_mae = absolute_or_relative_error(maxpooling_out.data, expected_max_pooling).mean()
    
    return maxpooling_mae < eps


@wrap_test(
    name='global avgpooling2d test',
    repeat=1000,
    meta={
        'description': 'Test global average pooling operation',
        'accepted_error': 1e-4
    }
)
def global_avg_pooling_correct_check():
    eps = 1e-4

    spatial_size = random.randint(16, 512)
    channel_in = random.randint(1, 512)
    
    t = Tensor.random_tensor([spatial_size, spatial_size, channel_in])  
    global_avgpooling_2d = keras.layers.GlobalAveragePooling2D()
    global_avgpooling_2d.build(t.data.reshape(1, *t.shape).shape)
    
    t_start = time.time()
    expected_global_avg_pooling = global_avgpooling_2d(t.data.reshape(1, *t.shape)).numpy().flatten()
    wrap_log('Tensorflow cpu', f'Elapsed time: {time.time() - t_start}')
    
    output = execute(Operation.GLOBAL_AVGPOOLING2D, [], t)
    diff = absolute_or_relative_error(output.data, expected_global_avg_pooling).mean()
    
    return diff <= eps
