import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from functools import partial


def Ax(rho, mask, g):
    return signal.convolve2d(rho, g, mode='same', )*mask
    # tmp = rho.copy()
    # tmp = np.pad(tmp, (0, tmp.shape[0]))
    # tmp_ft = np.fft.rfft2(tmp)
    # gtmp = np.pad(g, (0, tmp.shape[0]))
    # gft = np.fft.rfft2(gtmp)
    # tmp = np.fft.irfft2(tmp_ft*gft)
    # return tmp[:rho.shape[0], :rho.shape[1]]


def conjugate_gradient(x, b, func, niter):
    r = b - func(x)
    p = r.copy()
    rtr = np.inf
    count = 0
    while rtr > 1e-16:
        count += 1
        Ap = func(p)
        rtr = np.sum(p*r)
        alpha = rtr/np.sum(Ap*p)

        x = x + alpha*p
        rnew = r-alpha*Ap

        beta = np.sum(rnew*rnew)/rtr

        p = rnew+beta*p
        r = rnew
        print('on iteration ' + repr(count) + ' residual is ' + repr(rtr))
        if count > niter:
            break
    return x



def problem2b():
    niter=100
    n = 64
    side = 16
    nc = n//2

    xx, yy = np.mgrid[
        -nc:nc:((n+1)*1j),
        -nc:nc:((n+1)*1j)
        ]

    g = np.loadtxt('Results/greens.txt')
# 
    v = np.zeros((n+1, n+1))
    v[nc-side:nc+side, nc-side:nc+side] = 1

    mask = v > 0
    myfunc = partial(Ax, mask=mask, g=g)

    rho = np.zeros((n+1, n+1))

    rho = conjugate_gradient(rho, v, myfunc, niter=niter)
    np.savetxt('Results/rho.txt', rho)

    err = v - myfunc(rho)
    err = err[mask]
    avg_err = np.abs(err).mean()
    print(f'Average absolute error on the mask: {avg_err}')

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    ax[0].imshow(v)
    ax[0].set_title('Fixed Potential')

    ax[1].imshow(rho)
    im1 = ax[1].set_title('Charge Density')

    ax[2].imshow(myfunc(rho)[nc-side:nc+side, nc-side:nc+side])
    im1 = ax[2].set_title('Potential Inside Box \n from convolution')

    plt.tight_layout()
    plt.savefig('Results/2b1.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1, figsize=(3, 3))

    ax.plot(rho[nc-side, nc-side:nc+side])
    ax.set_title('Charge Density along side')

    plt.tight_layout()
    plt.savefig('Results/2b2.png')
    plt.clf()
    plt.close()



problem2b()