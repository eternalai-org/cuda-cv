import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from utils  import absolute_or_relative_error, log as wrap_log
from op import Operation, execute
from .test_registry import wrap_test

@wrap_test(
    name='add element wise test',
    repeat=1000,
    meta={
        'description': 'Test add element wise operation',
        'accepted_error': 1e-4
    }
)
def add_element_wise():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    t2 = Tensor.random_tensor(t1.shape)

    expected_add = t1.data + t2.data
    add_out = execute(Operation.ELEMENTWISE_ADD, [], [t1, t2])
    mae_add = absolute_or_relative_error(add_out.data, expected_add).mean()

    return mae_add <= accepted_error

@wrap_test(
    name='sub element wise test',
    repeat=1000,
    meta={
        'description': 'Test sub element wise operation',
        'accepted_error': 1e-4
    }
)
def sub_element_wise():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    t2 = Tensor.random_tensor(t1.shape)

    expected_sub = t1.data - t2.data
    sub_out = execute(Operation.ELEMENTWISE_SUB, [], [t1, t2])
    mae_sub = absolute_or_relative_error(sub_out.data, expected_sub).mean()

    return mae_sub <= accepted_error

@wrap_test(
    name='mul element wise test',
    repeat=1000,
    meta={
        'description': 'Test mul element wise operation',
        'accepted_error': 1e-4
    }
)
def mul_element_wise():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    t2 = Tensor.random_tensor(t1.shape)

    expected_mul = t1.data * t2.data
    mul_out = execute(Operation.ELEMENTWISE_MUL, [], [t1, t2])
    mae_mul = absolute_or_relative_error(mul_out.data, expected_mul).mean()

    return mae_mul <= accepted_error

@wrap_test(
    name='div element wise test',
    repeat=1000,
    meta={
        'description': 'Test div element wise operation',
        'accepted_error': 1e-4
    }
)
def div_element_wise():
    accepted_error = 1e-4
    t1 = Tensor.random_tensor()
    t2 = Tensor.random_tensor(t1.shape)

    expected_div = t1.data / t2.data
    div_out = execute(Operation.ELEMENTWISE_DIV, [], [t1, t2])
    mae_div = absolute_or_relative_error(div_out.data, expected_div).mean()

    return mae_div <= accepted_error
