from matplotlib import pyplot as plt

__que = []

def push(op, rel, x, y):
    __que.append(f'{op} {rel} {x} {y}')

def flush():
    with open('stats.txt', 'a') as f:
        f.write('\n'.join(__que) + '\n')

def reload():
    with open('stats.txt', 'r') as f:
        data = f.read().splitlines()

    ops = {}
    
    for d in data:
        op, rel, x, y = d.split(' ')
        k = f'{op}:{rel}'

        if op not in ops:
            ops[k] = []

        ops[k].append((int(x), int(y)))

    return ops

def make_report():
    data = reload()

    categories = {
        k: k.split(':')[0] 
        for k in data.keys()
    }

    for k, v in categories.items():
        x, y = zip(*data[k])
        plt.plot(x, y, label=v)