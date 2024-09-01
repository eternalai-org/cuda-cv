import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
from tensor import Tensor
from op import execute, Operation
import tensorflow as tf
import time
from utils import log as wraplog, absolute_or_relative_error
from .test_registry import wrap_test

@wrap_test(
    name='conv2d correct random test', 
    repeat=1000, 
    meta={
        'description': 'Test conv2d operation, execution time includes tensorflow operation on cpu',
        'accepted_error': 1e-4
    }
)
def test_correct_conv2d(*args):
    eps = 1e-4

    spatial_size = random.randint(8, 512)
    channel_in = random.randint(1, 512)
    channel_out = random.randint(1, 512)

    t = Tensor.random_tensor([spatial_size, spatial_size, channel_in])

    ksize = random.randint(1, min(8, spatial_size))
    stride = random.randint(1, ksize)
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
    wraplog('Tensorflow cpu', f'Elapsed time: {time.time() - t_start}')

    conv2d_out = execute(Operation.CONV2D, params, [t, random_kernel, random_bias])
    conv2d_mae = absolute_or_relative_error(conv2d_out.data, expected_conv2d).mean()
    return conv2d_mae <= eps    
