import numpy as np
from utils import compress_uint256, unpack_uint256, chunked

class Tensor(object):
    def __init__(self, data, shape) -> None:
        self._data = data
        self._shape = shape
        
        if not isinstance(data, np.ndarray):
            self._data = np.array(data)
            
        self._data = self._data.flatten()

    @property
    def data(self): return self._data

    @property
    def shape(self): return self._shape

    def compress(self):
        return [compress_uint256(*i) for i in chunked(self._data.flatten().tolist(), 4)]
    
    @staticmethod
    def uncompress(b: list, shape = None) -> 'Tensor':
        data = [unpack_uint256(i) for i in b]
        data = np.array(data, dtype=np.int64)

        if shape is not None:
            data = data.flatten()[:np.prod(shape)] # remove padding
            data = data.reshape(shape)
        else:
            data = data.flatten()

        return Tensor(data, shape)
