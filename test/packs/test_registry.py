from utils import log
import time
import enlighten

__test_fn_registry = []

def wrap_test(name, meta={}, repeat=1):
    def wrapper(fn):
        __test_fn_registry.append((name, meta, fn, repeat))
        return fn
    return wrapper

def repeat(fn, n):
    for _ in range(n):
        yield fn()

def run_tests(targets=[]):
    manager = enlighten.get_manager()
    pbar_global = manager.counter(total=len(__test_fn_registry), desc='Testing progress', unit='tests')
    status_format = '{program}{fill}{stage}{fill}{status}'

    pstatus = manager.status_bar(
        status_format=status_format,
        color='bold_slategray',
        program='Testing',
        stage='Preparing',
        status='Starting'
    )
    
    total_fails, total_executed = 0, 0

    for i, (test_name, meta, fn, rp) in enumerate(__test_fn_registry):
        if len(targets) > 0 and test_name not in targets:
            continue
        
        fails, executed = 0, 0

        log(f'--- Test meta: ')

        for k, v in meta.items():
            log(f'  - {k}: {v}')

        pbar_local = manager.counter(total=rp, desc=f'Test {test_name}', unit='iterations')

        try:
            for i, res in enumerate(repeat(fn, rp)):
                executed += 1
                pbar_local.update()
                
                if not res:
                    log(f'Test {test_name} at iteration {i} failed.')
                    fails += 1

                if i % 10 == 0:
                    pstatus.update(stage=f'Test {test_name}', status=f'Total fails: {total_fails + fails}/{total_executed + i}')

        except KeyboardInterrupt:
            log('Interrupted')

        pbar_local.close()
        log(f'--- Verdict: {"PASSED" if fails == 0 else "FAILED"}')

        total_fails += fails
        total_executed += executed
        
        pbar_global.update()

def run_benchmarks():
    manager = enlighten.get_manager()
    
    pbar_global = manager.counter(total=len(__test_fn_registry), desc='Testing rogress', unit='tests')
    
    test_results = {}
    for i, (test_name, meta, fn, rp) in enumerate(__test_fn_registry):
        
        time_consumed, executed = 0, 0
        
        log(f'--- Test meta: ')

        for k, v in meta.items():
            log(f'  - {k}: {v}')

        trace = time()

        pbar_local = manager.counter(total=rp, desc=f'Test {test_name}', unit='iterations')

        try:
            for i, res in enumerate(repeat(fn, rp)):
                executed += 1                
                pbar_local.update( )

        except KeyboardInterrupt:
            log('*Interrupted')
        
        pbar_local.close()
        time_consumed = time() - trace

        test_results[i] = {
            'name': test_name,
            'meta': meta,
            'time': time_consumed,
            'executed': executed
        }
        
        pbar_global.update()
        
def list_names():
    for i, (test_name, meta, fn, rp) in enumerate(__test_fn_registry):
        yield test_name