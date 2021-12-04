import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from functools import partial


def problem2c(n, side, edge):
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

    v = signal.convolve2d(rho, g, mode='same')

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
    mag_grad = mag_grad[edge:-edge, edge:-edge]
    plt.title('Magnitude of the Electric Field')
    plt.imshow(mag_grad)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('Results/2c2.png')
    plt.clf()
    plt.close()

    dx = np.ma.masked_where(mask, dx)
    dy = np.ma.masked_where(mask, dy)

    dx, dy = dx[edge:-edge, edge:-edge], dy[edge:-edge, edge:-edge]
    xx, yy = xx[edge:-edge, edge:-edge], yy[edge:-edge, edge:-edge]

    skip = (slice(None, None, 3), slice(None, None, 3))

    fig, ax = plt.subplots()
    extent = nc - edge
    extent = (-extent, extent, -extent, extent)
    masked_v = np.ma.masked_where(mask, v)
    ax.quiver(xx[skip], yy[skip], dx[skip], dy[skip], scale=0.25)
    ax.imshow(v[edge:-edge, edge:-edge], extent=extent)
    plt.contour(masked_v[edge:-edge, edge:-edge], colors='white',
                extent=extent)
    ax.set_title('Electric Field Lines')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('Results/2c3.png')
    plt.clf()
    plt.close()
