import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
from tensor import Tensor
from op import execute, Operation
from utils import sigmoid, softmax, tanh, relu, mae, absolute_or_relative_error

def run_case(*args):
    accepted_error = 1e-3

    sigmoid_opcode = Operation.SIGMOID
    softmax_opcode = Operation.SOFTMAX
    tanh_opcode = Operation.TANH
    relu_opcode = Operation.RELU
    
    tin = Tensor.random_tensor([10])
    
    # tin._data += 0.5
    
    expected_sigmoid = sigmoid(tin.data)
    expected_softmax = softmax(tin.data)
    expected_tanh = tanh(tin.data)
    expected_relu = relu(tin.data)

    softmax_out = execute(softmax_opcode, [], tin)
    tanh_out = execute(tanh_opcode, [], tin)
    sigmoid_out = execute(sigmoid_opcode, [], tin)
    relu_out = execute(relu_opcode, [], tin)
    
    if not sigmoid_out or not softmax_out or not tanh_out or not relu_out:
        return False
    
    sigmoid_mae = mae(expected_sigmoid, sigmoid_out.data)
    softmax_mae = absolute_or_relative_error(softmax_out.data, expected_softmax).mean()
    tanh_mae = mae(expected_tanh, tanh_out.data)
    relu_mae = mae(expected_relu, relu_out.data)
        
    return not(sigmoid_mae > accepted_error \
        or softmax_mae > accepted_error \
        or tanh_mae > accepted_error \
        or relu_mae > accepted_error)    

def benchmark_activation():
    n_cases = 1000

    futures = []
    
    try:
        for _ in tqdm(range(n_cases), total=n_cases, desc='Running test cases'):
            futures.append(run_case())
    except KeyboardInterrupt:
        print('Interrupted')

    fails = sum([f == False for f in tqdm(futures, total=len(futures), desc='Processing results')])
    success = len(futures) - fails

    print(f'Success: {success}/{len(futures)}')
    print(f'Fails: {fails}/{len(futures)}')

    if fails > 0:
        raise ValueError('Some test cases failed')

if __name__ == '__main__':
    benchmark_activation()