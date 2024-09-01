import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
from tensor import Tensor
from op import execute
from utils import compare_tensors, log as wrap_log
from .test_registry import wrap_test

@wrap_test(
    name='abi random test', 
    repeat=1000, 
    meta={
        'description': 'Test ABI encode/decode with random tensor'
    }
)
def run_case(*args):
    sample = Tensor.random_tensor()
    tout = execute(27, [], [sample])

    if tout is None or not (compare_tensors(sample.data, tout.data) and compare_tensors(sample.shape, tout.shape)):
        wrap_log('ABI encode/decode has failed with shape', sample.shape)
        return False

    return True


@wrap_test(
    name='abi zero test', 
    repeat=1000, 
    meta={
        'description': 'Test ABI encode/decode with zero tensor'
    }
)
def run_case(*args):
    sample = Tensor.random_tensor()
    tout = execute(27, [], [sample])

    if tout is None or not (compare_tensors(sample.data, tout.data) and compare_tensors(sample.shape, tout.shape)):
        wrap_log('ABI encode/decode (zero tensor) has failed with shape', sample.shape)
        return False

    return True