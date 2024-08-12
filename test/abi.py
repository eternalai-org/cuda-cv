from eth_abi import encode as abi_encode, decode as abi_decode
from random import randint
import numpy as np

# uing64_t, uint64_t[], uint64_t[][], uint256[][]

def generate_tensor():
    ndim = randint(1, 5)
    shape = [randint(1, 6) for _ in range(ndim)]
    flattened = (np.prod(shape) + 3) // 4 # padded to 32 bytes
    return (np.random.rand(flattened) * 100).astype(np.uint64), shape

def random_opcode():
    return randint(0, 10)

def generate_sample():
    n_tensors = randint(1, 3)
    tensors_shapes = [generate_tensor() for _ in range(n_tensors)]

    tensors = [t[0].tolist() for t in tensors_shapes] # uint256[][]
    shapes = [t[1] for t in tensors_shapes] # uint64_t[]
    opcode = random_opcode() # uint64_t
    random_params = (np.random.rand(randint(0, 10)) * 100).astype(np.uint64).tolist() # uint256[]
    
    return opcode, random_params, shapes, tensors


sample = generate_sample()
print(sample)
encoded = abi_encode(('uint64', 'uint64[]', 'uint64[][]', 'uint256[][]'), sample)

with open('data.bin', 'wb') as f:
    print(f.write(encoded), 'bytes written')

def chunked(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

for l in chunked(encoded, 32):
    print(int.from_bytes(l, 'big'))