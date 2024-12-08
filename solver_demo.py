import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from cube import Cube
from get_state import get_state



state = get_state()
cube = Cube("QTMIX")
cube.set_state(state)

def visualize(state):
    grid = [[-1, -1, -1,  2, 5, 8,  -1, -1, -1,  -1, -1, -1],
            [-1, -1, -1,  1, 4, 7,  -1, -1, -1,  -1, -1, -1],
            [-1, -1, -1,  0, 3, 6,  -1, -1, -1,  -1, -1, -1],
            [20, 23, 26,  47, 50, 53,  29, 32, 35,  38, 41, 44],
            [19, 22, 25,  46, 49, 52,  28, 31, 34,  37, 40, 43],
            [18, 21, 24,  45, 48, 51,  27, 30, 33,  36, 39, 42],
            [-1, -1, -1,  11, 14, 17,  -1, -1, -1,  -1, -1, -1],
            [-1, -1, -1,  10, 13, 16,  -1, -1, -1,  -1, -1, -1],
            [-1, -1, -1,  9, 12, 15,  -1, -1, -1,  -1, -1, -1]]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, 12)
    ax.set_ylim(-8, 1)
    plt.axis('off')
    for i in range(9):
        for j in range(12):
            if grid[i][j] == -1:
                continue
            color = {0: 'yellow', 1: 'white', 2: 'blue',
                    3: 'green', 4: 'orange', 5: 'red'}[state[grid[i][j]]]
            square = patches.Rectangle(
                (j, -i), 1, 1, edgecolor='black', facecolor=color)
            ax.add_patch(square)
    plt.show()

visualize(cube.get_state())

