from matplotlib import pyplot as plt
import os
import numpy as np

__que = []

def push(op, rel, x, y):
    __que.append(f'{op} {rel} {x} {y}')

def flush():
    global __que
    with open('stats.txt', 'a') as f:
        f.write('\n'.join(__que) + '\n')
    __que.clear()

def reload():
    with open('stats.txt', 'r') as f:
        data = f.read().splitlines()
        
    try: os.remove('stats.txt')
    except: pass

    ops = {}
    
    for d in data:
        op, rel, x, y = d.split(' ')
        k = f'{op}:{rel}'

        if op not in ops:
            ops[k] = []

        ops[k].append((float(x), float(y)))

    return ops

def make_report():
    data = reload()

    categories = {}
    
    for x in data:
        c = x.split(':')[0]

        if c not in categories:
            categories[c] = []
            
        categories[c].append(x)
    
    scatter_w, scatter_h, padding = 4, 4, 1
    
    y_scatters = len(categories)
    x_scatters = max([len(v) for v in categories.values()])

    fig_w = scatter_w * x_scatters + padding * (x_scatters - 1)
    fig_h = scatter_h * y_scatters + padding * (y_scatters - 1)

    fig, axs = plt.subplots(y_scatters, x_scatters, figsize=(fig_w, fig_h))

    for i, (cat, values) in enumerate(categories.items()):
        for j, k in enumerate(values):
            d = np.array(data[k])
            x, y = d[:, 0], d[:, 1]
            
            print(i, j, x, y)

            axs[i, j].scatter(x, y)
            axs[i, j].set_title(k)

    fig.tight_layout()
    
    # to file
    plt.savefig('stats.png')
    
if __name__ == '__main__':
    push('a', 'b', 1, 2)
    push('a', 'b', 3, 4)
    push('a', 'b', 5, 6)
    push('a', 'c', 7, 8)
    push('a', 'c', 9, 10)
    push('a', 'c', 11, 12)
    
    push('a', 'b', 2, 2)
    push('a', 'b', 3, 5)
    push('a', 'b', 5, 5)
    push('a', 'c', 7, 5)
    push('a', 'c', 9, 5)
    push('a', 'c', 11, 5)
    
    push('e', 'b', 1, 2)
    push('e', 'b', 3, 4)
    push('e', 'b', 5, 6)
    push('e', 'c', 7, 8)
    push('e', 'c', 9, 10)
    push('e', 'c', 11, 12)
    
    push('e', 'b', 2, 2)
    push('e', 'b', 3, 5)
    push('e', 'b', 5, 5)
    push('e', 'c', 7, 5)
    push('e', 'c', 9, 5)
    push('e', 'c', 11, 5)
    
    flush()
    make_report()