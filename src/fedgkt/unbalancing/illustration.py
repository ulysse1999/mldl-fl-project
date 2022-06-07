import numpy as np

# understand the effect of \alpha, the concentration parameter of Dirichlet distribution

N_CLIENTS=20
N_CLASSES=10
alpha=0.5

s = np.random.dirichlet([alpha]*N_CLASSES, N_CLIENTS).transpose()

import matplotlib.pyplot as plt

class_to_color = {
    0 : 'tab:blue',
    1 : 'tab:orange',
    2 : 'tab:green',
    3 : 'tab:red',
    4 : 'tab:purple',
    5 : 'tab:brown',
    6 : 'tab:pink',
    7 : 'tab:gray',
    8  : 'tab:olive',
    9 : 'tab:cyan'
}

for i in range(N_CLASSES):
    if i==0:
        plt.barh(range(N_CLIENTS), s[0], color=class_to_color[i])
    else:
        plt.barh(range(N_CLIENTS), s[i], left=sum([s[j] for j in range(i)]), color=class_to_color[i])

