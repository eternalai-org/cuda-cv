import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from utils  import absolute_or_relative_error, log as wrap_log
from op import Operation, execute
from .test_registry import wrap_test

@wrap_test(
    name='array sqrt test',
    repeat=1000,
    meta={
        'description': 'Test array sqrt transformation operation',
        'accepted_error': 1e-4
    }
)
def array_sqrt():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    t1._data += 0.5
    expected_sqrt = np.sqrt(t1.data)
    add_out = execute(Operation.TRANSFORM_SQRT, [], [t1])
    mae_add = absolute_or_relative_error(add_out.data, expected_sqrt).mean()
    return mae_add <= accepted_error


@wrap_test(
    name='array exp test',
    repeat=1000,
    meta={
        'description': 'Test array exp transformation operation',
        'accepted_error': 1e-4
    }
)
def array_exp():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    expected_exp = np.exp(t1.data)
    add_out = execute(Operation.TRANSFORM_EXP, [], [t1])
    mae_add = absolute_or_relative_error(add_out.data, expected_exp).mean()
    return mae_add <= accepted_error


@wrap_test(
    name='array max/min scale test',
    repeat=1000,
    meta={
        'description': 'Test array max/min scale transformation operation',
        'accepted_error': 1e-4
    }
)
def array_max_min_scale():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    _min, _max = t1.data.min(), t1.data.max()
    expected_max_min_scale = (t1.data - _min) / (_max - _min)
    add_out = execute(Operation.MIN_MAX_SCALE, [], [t1])
    mae_add = absolute_or_relative_error(add_out.data, expected_max_min_scale).mean()
    return mae_add <= accepted_error


@wrap_test(
    name='array zscore test',
    repeat=1000,
    meta={
        'description': 'Test array zscore transformation operation',
        'accepted_error': 1e-4
    }
)
def array_zscore():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    _mean, _std = t1.data.mean(), t1.data.std()
    eps = 1e-4
    expected_zscore = (t1.data - _mean) / (_std + eps)
    add_out = execute(Operation.ZSCORE, [int(eps * (1 << 32))], [t1])
    mae_add = absolute_or_relative_error(add_out.data, expected_zscore).mean()
    return mae_add <= accepted_error