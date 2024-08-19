def compress_uint256(a1 = 0, a2 = 0, a3 = 0, a4 = 0):
        return a4 + (a3 << 64) + (a2 << 128) + (a1 << 192)
    
def unpack_uint256(a):
    a1 = (a >> 192) & 0xFFFFFFFFFFFFFFFF
    a2 = (a >> 128) & 0xFFFFFFFFFFFFFFFF
    a3 = (a >> 64) & 0xFFFFFFFFFFFFFFFF
    a4 = a & 0xFFFFFFFFFFFFFFFF
    
    return a1, a2, a3, a4

def chunked(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]