import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from utils  import absolute_or_relative_error, log as wrap_log
from op import Operation, execute
from .test_registry import wrap_test
import os

@wrap_test(
    name='sum reduction test',
    repeat=1000,
    meta={
        'description': 'Test sum reduction operation',
        'accepted_error': 1e-4
    }
)
def sum_reduction():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    expected_sum = np.sum(t1.data)
    add_out = execute(Operation.REDUCTION_SUM, [], [t1])
    mae_add = absolute_or_relative_error(add_out.data, expected_sum).mean()
    return mae_add <= accepted_error


@wrap_test(
    name='mean reduction test',
    repeat=1000,
    meta={
        'description': 'Test mean reduction operation',
        'accepted_error': 1e-4
    }
)
def mean_reduction():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    expected_mean = np.mean(t1.data)
    add_out = execute(Operation.REDUCTION_MEAN, [], [t1])
    mae_add = absolute_or_relative_error(add_out.data, expected_mean).mean()
    return mae_add <= accepted_error


@wrap_test(
    name='max reduction test',
    repeat=1000,
    meta={
        'description': 'Test max reduction operation',
        'accepted_error': 1e-4
    }
)
def max_reduction():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    expected_max = np.max(t1.data)
    add_out = execute(Operation.REDUCTION_MAX, [], [t1])
    mae_add = absolute_or_relative_error(add_out.data, expected_max).mean()
    return mae_add <= accepted_error

@wrap_test(
    name='min reduction test',
    repeat=1000,
    meta={
        'description': 'Test min reduction operation',
        'accepted_error': 1e-4
    }
)
def min_reduction():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    expected_min = np.min(t1.data)
    add_out = execute(Operation.REDUCTION_MIN, [], [t1])
    mae_add = absolute_or_relative_error(add_out.data, expected_min).mean()
    return mae_add <= accepted_error
    