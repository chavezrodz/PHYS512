import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a/(x-b)**2 + c


def problem_6():
    niters = 1000
    n_sims = 20

    walks = np.array([np.cumsum(np.random.randn(niters)) for i in range(n_sims)])
    # shape = walks x steps

    fft = np.square(np.abs(
                 np.fft.rfft(walks, axis=1)[:, 1:]
                 )).mean(axis=0)

    xdata = np.arange(niters//2)
    popt, pcov = curve_fit(func, xdata, fft, p0=[8e7, -1.5, 10])

    fit = func(xdata, *popt)
    err = (fft - fit)
    std = err.std()

    fig, ax = plt.subplots(3)

    ax[0].plot(walks.T)
    ax[0].set_ylabel('Position')

    ax[1].set_ylabel('Power Spectrum')
    ax[1].plot(fft, '.', label='Data')
    ax[1].plot(xdata, fit, 'r-', label='fit')
    ax[1].legend()

    ax[2].set_ylabel('Residuals')
    ax[2].scatter(xdata, err, marker='.')
    plt.savefig('Results/q6.png')
    plt.clf()
    plt.close()


    print(
        f"""
        Problem 6)
        b)
        We fit the power spectrum of the random walk with 1/k^2
        and achieve an average error of {err.mean():.3e} 
        """)
