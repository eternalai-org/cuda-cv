import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tensor import Tensor
from op import Operation, execute
import traceback
from utils import log as wrap_log
from .test_registry import wrap_test

@wrap_test(
    name='concatenate test',
    repeat=1000,
    meta={
        'description': 'Test concatenate operation',
    }
)
def concatenate_test(): 
    t1 = Tensor.random_tensor()
    t2 = Tensor.random_tensor(t1.shape)
    t3 = Tensor.random_tensor(t1.shape)
    
    t1_data, t2_data, t3_data = t1.data.reshape(t1.shape), t2.data.reshape(t2.shape), t3.data.reshape(t3.shape)
    axis = np.random.randint(0, len(t1.shape))
    
    expected = np.concatenate([t1_data, t2_data, t3_data], axis=axis)
    params = [axis]
    
    out = execute(Operation.CONCATENATE, params, [t1, t2, t3])
    return np.allclose(out.data.reshape(out.shape), expected)
