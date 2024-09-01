import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from op import Operation, execute
import random
from keras import layers
from utils import absolute_or_relative_error, to_i64, log as wrap_log
import os
import time
from .test_registry import wrap_test

@wrap_test(
    name='batch norm test',
    repeat=1000,
    meta={
        'description': 'Test batch norm operation',
        'accepted_error': 1e-4
    }
)
def batch_norm_test(): 
    accepted_error = 1e-4

    spatial_size = random.randint(8, 128)
    channel_in = random.randint(1, 64)

    t1 = Tensor.random_tensor([spatial_size, spatial_size, channel_in])
    t1_data = t1.data.reshape(t1.shape)

    ma, mv, beta, gama = Tensor.random_tensor([channel_in]), Tensor.random_tensor([channel_in]), \
        Tensor.random_tensor([channel_in]), Tensor.random_tensor([channel_in])
        
    mv._data += 0.5

    batch_norm_layer = layers.BatchNormalization()
    batch_norm_layer.build(t1_data.shape)
    batch_norm_layer.set_weights([gama.data, beta.data, ma.data, mv.data])
    eps = to_i64(int(1e-3 * 2 ** 32))
    
    t_start = time.time()
    expected = batch_norm_layer(t1_data).numpy().flatten()
    wrap_log('Tensorflow cpu', f'Elapsed time: {time.time() - t_start}')
    
    actual = execute(Operation.BATCH_NORM, [eps], [t1, gama, beta, ma, mv])
    err = absolute_or_relative_error(actual.data, expected).mean()

    return err <= accepted_error
