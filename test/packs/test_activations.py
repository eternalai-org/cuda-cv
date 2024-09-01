from tensor import Tensor
from op import execute, Operation
from utils import sigmoid, softmax, tanh, relu, mae, absolute_or_relative_error, log as wrap_log
from .test_registry import wrap_test

@wrap_test(
    name='sigmoid activations test',
    repeat=1000,
    meta={
        'description': 'Test sigmoid activation function',
        'accepted_error': 1e-4
    }
)
def test_sigmoid():
    accepted_error = 1e-4
    sigmoid_opcode = Operation.SIGMOID
    
    tin = Tensor.random_tensor()
    tin._data += 0.5

    expected_sigmoid = sigmoid(tin.data)
    sigmoid_out = execute(sigmoid_opcode, [], tin)    
    sigmoid_mae = absolute_or_relative_error(expected_sigmoid, sigmoid_out.data).mean()

    return sigmoid_mae <= accepted_error  


@wrap_test(
    name='softmax activations test',
    repeat=1000,
    meta={
        'description': 'Test softmax activation function',
        'accepted_error': 1e-4
    }
)
def test_softmax():
    accepted_error = 1e-4
    softmax_opcode = Operation.SOFTMAX
    
    tin = Tensor.random_tensor()
    tin._data += 0.5

    expected_softmax = softmax(tin.data)
    softmax_out = execute(softmax_opcode, [], tin)    
    softmax_mae = absolute_or_relative_error(expected_softmax, softmax_out.data).mean()

    return softmax_mae <= accepted_error  


@wrap_test(
    name='tanh activations test',
    repeat=1000,
    meta={
        'description': 'Test tanh activation function',
        'accepted_error': 1e-4
    }
)
def test_tanh():
    accepted_error = 1e-4
    tanh_opcode = Operation.TANH

    tin = Tensor.random_tensor()
    tin._data += 0.5

    expected_tanh = tanh(tin.data)
    tanh_out = execute(tanh_opcode, [], tin)    
    tanh_mae = absolute_or_relative_error(expected_tanh, tanh_out.data).mean()

    return tanh_mae <= accepted_error


@wrap_test(
    name='relu activations test',
    repeat=1000,
    meta={
        'description': 'Test relu activation function',
        'accepted_error': 1e-4
    }
)
def test_relu():
    accepted_error = 1e-4
    relu_opcode = Operation.RELU

    tin = Tensor.random_tensor()
    tin._data += 0.5

    expected_relu = relu(tin.data)
    relu_out = execute(relu_opcode, [], tin)    
    relu_mae = absolute_or_relative_error(expected_relu, relu_out.data).mean()

    return relu_mae <= accepted_error
