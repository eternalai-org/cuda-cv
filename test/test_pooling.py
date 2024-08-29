import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import keras
import random
from tensor import Tensor
from op import Operation, execute
from utils import absolute_or_relative_error

def run_case(*args): 
    eps = 1e-4

    spatial_size = random.randint(16, 256)
    channel_in = random.randint(1, 256)
    
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
    
    avgpooling = keras.layers.AveragePooling2D(
        pool_size=(window_size, window_size),
        strides=(stride, stride),
        padding=padding
    )
    
    globalAvgPooling = keras.layers.GlobalAveragePooling2D()
    
    params = [window_size, window_size, stride, stride, padding_i]
    expected_max_pooling = maxpooling(t.data.reshape(1, *t.shape)).numpy().flatten()
    expected_avg_pooling = avgpooling(t.data.reshape(1, *t.shape)).numpy().flatten()
    expected_global_avg_pooling = globalAvgPooling(t.data.reshape(1, *t.shape)).numpy().flatten()

    maxpooling_out = execute(Operation.MAXPOOLING2D, params, t)
    avgpooling_out = execute(Operation.AVGPOOLING2D, params, t)
    global_avg_pooling_out = execute(Operation.GLOBAL_AVGPOOLING2D, [], t)
    
    maxpooling_mae = absolute_or_relative_error(maxpooling_out.data, expected_max_pooling).mean()
    avgpooling_mae = absolute_or_relative_error(avgpooling_out.data, expected_avg_pooling).mean()
    global_avg_pooling_mae = absolute_or_relative_error(global_avg_pooling_out.data, expected_global_avg_pooling).mean()
    
    res = all([
        maxpooling_mae < eps,
        avgpooling_mae < eps,
        global_avg_pooling_mae < eps
    ])
    
    if not res:
        print(padding)
        print(f'MaxPooling MAE: {maxpooling_mae}')
        print(f'AvgPooling MAE: {avgpooling_mae}')
        print(f'GlobalAvgPooling MAE: {global_avg_pooling_mae}')
    
    return res
    
def benchmark_pooling():
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
    

def run_case_special_1(*args): 
    eps = 1e-4

    spatial_size = random.randint(4, 9)
    channel_in = 1024 # random.randint(1, 4096)

    t = Tensor.random_tensor([spatial_size, spatial_size, channel_in])
    window_size = random.randint(1, min(16, spatial_size))
    stride = random.randint(1, window_size)
    padding = random.choice(['valid', 'same'])

    padding_i = 1 if padding == 'same' else 0
    # maxpooling = keras.layers.MaxPooling2D(
    #     pool_size=(window_size, window_size), 
    #     strides=(stride, stride), 
    #     padding=padding
    # )
    
    # avgpooling = keras.layers.AveragePooling2D(
    #     pool_size=(window_size, window_size),
    #     strides=(stride, stride),
    #     padding=padding
    # )
    
    globalAvgPooling = keras.layers.GlobalAveragePooling2D()
    
    params = [window_size, window_size, stride, stride, padding_i]
    # expected_max_pooling = maxpooling(t.data.reshape(1, *t.shape)).numpy().flatten()
    # expected_avg_pooling = avgpooling(t.data.reshape(1, *t.shape)).numpy().flatten()
    expected_global_avg_pooling = globalAvgPooling(t.data.reshape(1, *t.shape)).numpy().flatten()

    # maxpooling_out = execute(Operation.MAXPOOLING2D, params, t)
    # avgpooling_out = execute(Operation.AVGPOOLING2D, params, t)
    global_avg_pooling_out = execute(Operation.GLOBAL_AVGPOOLING2D, [], t)
    
    # maxpooling_mae = absolute_or_relative_error(maxpooling_out.data, expected_max_pooling).mean()
    # avgpooling_mae = absolute_or_relative_error(avgpooling_out.data, expected_avg_pooling).mean()
    global_avg_pooling_mae = absolute_or_relative_error(global_avg_pooling_out.data, expected_global_avg_pooling).mean()
    
    res = all([
        # maxpooling_mae < eps,
        # avgpooling_mae < eps,
        global_avg_pooling_mae < eps
    ])
    
    if not res:
        print(padding)
        print(f'MaxPooling MAE: {maxpooling_mae}')
        print(f'AvgPooling MAE: {avgpooling_mae}')
        print(f'GlobalAvgPooling MAE: {global_avg_pooling_mae}')
    
    return res
    
def benchmark_pooling_special_1():
    n_cases = 1000

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
    # benchmark_pooling()
    benchmark_pooling_special_1()