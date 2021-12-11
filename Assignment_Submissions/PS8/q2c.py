import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from functools import partial
from q2b import Ax


def problem2c(n, side):
    rho = np.loadtxt('Results/rho.txt')

    nc = n//2
    v = np.zeros((n+1, n+1))
    v[nc-side:nc+side, nc-side:nc+side] = 1

    mask = v > 0

    xx, yy = np.mgrid[
        -nc:nc:((n+1)*1j),
        -nc:nc:((n+1)*1j)
        ]

    g = np.loadtxt('Results/greens.txt')

    v = Ax(rho, g)

    plt.title('Potential Everywhere')    
    plt.imshow(v)
    plt.colorbar()
    plt.contour(v, colors='white')
    plt.tight_layout()
    plt.savefig('Results/2c1.png')
    plt.clf()
    plt.close()

    dx, dy = np.gradient(v)
    mag_grad = np.sqrt(dx**2 + dy**2)
    plt.title('Magnitude of the Electric Field')
    plt.imshow(mag_grad)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('Results/2c2.png')
    plt.clf()
    plt.close()

    dx = np.ma.masked_where(mask, dx)
    dy = np.ma.masked_where(mask, dy)

    skip = (slice(None, None, 3), slice(None, None, 3))

    fig, ax = plt.subplots()
    extent = nc
    extent = (-extent, extent, -extent, extent)
    masked_v = np.ma.masked_where(mask, v)
    ax.quiver(xx[skip], yy[skip], dx[skip], dy[skip], scale=0.25)
    ax.imshow(v, extent=extent)
    plt.contour(masked_v, colors='white', extent=extent)
    ax.set_title('Electric Field Lines')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('Results/2c3.png')
    plt.clf()
    plt.close()
