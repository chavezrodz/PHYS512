import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from functools import partial


# Problem a
def potential(x, y):
    r = np.sqrt(x**2 + y**2)
    return -np.log(r)/(2*np.pi)


def average(x):
    # Ignoring Edge effects
    avg = (
        np.roll(x, 1, axis=0) +
        np.roll(x, -1, axis=0) +
        np.roll(x, 1, axis=1) +
        np.roll(x, -1, axis=1)
        )/4
    return avg


def problem2a(n):
    nc = n//2
    xx, yy = np.mgrid[
        -nc:nc:((n+1)*1j),
        -nc:nc:((n+1)*1j)
        ]

    v = potential(xx, yy)

    # Origin average of neighbors
    v[nc, nc] = v[nc+1, nc] - (v[nc+2, nc] + v[nc+1, nc+1] + v[nc+1, nc-1])

    # rescaling

    rho = v - average(v)
    v = v/rho[nc, nc]
    rho = v - average(v)
    v = v - v[nc, nc] + 1

    np.savetxt('Results/greens.txt', v)

    print(f'\nPotential at origin: {v[nc, nc]}\n'
          f'Density at origin: {rho[nc, nc]}\n'
          f'\n'

          f'Potential at [1, 0]: {v[nc + 1, nc]}\n'
          f'Potential at [2, 0]: {v[nc + 2, nc]}\n'
          f'\n'
          f'Potential at [5, 0]: {v[nc + 5, nc]}\n'
          )

    plt.imshow(v)
    plt.title('Potential from single point charge')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('Results/2a1.png')
    plt.clf()
    plt.close()