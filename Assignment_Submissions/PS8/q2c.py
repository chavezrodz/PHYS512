import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from functools import partial


def problem2c():
    rho = np.loadtxt('Results/rho.txt')
    # print(rho.shape)
    n = 64
    side = 16
    edge = 5
    nc = n//2
    plt.plot(rho[nc-side, nc-side:nc+side])
    plt.show()

    assert False

    v = np.zeros((n+1, n+1))
    v[nc-side:nc+side, nc-side:nc+side] = 1

    mask = v > 0

    xx, yy = np.mgrid[
        -nc:nc:((n+1)*1j),
        -nc:nc:((n+1)*1j)
        ]

    g = np.loadtxt('Results/greens.txt')

    v = signal.convolve2d(rho, g, mode='same')

    plt.title('Potential Everywhere')    
    plt.imshow(v, )
    plt.colorbar()
    plt.contour(v, levels=[0.3, 0.5, 0.8], colors='white')
    plt.savefig('Results/2c1.png')
    plt.clf()
    plt.close()

    dx, dy = np.gradient(v)
    mag_grad = np.sqrt(dx**2 + dy**2)

    plt.title('Magnitude of the Electric Field')
    plt.imshow(mag_grad)
    plt.colorbar()
    plt.savefig('Results/2c2.png')
    plt.clf()
    plt.close()


    # dx[mask], dy[mask] = 0, 0
    dx = np.ma.masked_where(mask, dx)
    dy = np.ma.masked_where(mask, dy)
    dx, dy = dx[edge:-edge, edge:-edge], dy[edge:-edge, edge:-edge]

    fig, ax = plt.subplots()
    ax.quiver(dx, dy, scale=0.5)
    ax.set_title('Electric Field Lines')
    ax.set_aspect('equal')
    plt.savefig('Results/2c3.png')
    plt.show()
    plt.clf()
    plt.close()

problem2c()