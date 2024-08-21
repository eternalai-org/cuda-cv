import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def run_case(*args): 
    return True

def benchmark_concatenate():
    n_cases = 10 * 1000

    futures = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for _ in tqdm(range(n_cases), total=n_cases, desc='Running test cases'):
            futures.append(executor.submit(run_case))

    fails = sum([not f.result() for f in futures])
    success = n_cases - fails

    print(f'Success: {success}/{n_cases}')
    print(f'Fails: {fails}/{n_cases}')

    if fails > 0:
        raise ValueError('Some test cases failed')

if __name__ == '__main__':
    benchmark_concatenate()