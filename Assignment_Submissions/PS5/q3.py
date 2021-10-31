import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu=0, sigma=1):
    return np.exp(-((x-mu)/(2*sigma))**2)


def shift_fft(arr, shift):
    n_t = len(arr)
    f = np.arange(n_t)/n_t
    rhs = np.exp(-2j*np.pi*f*shift)
    conv = np.fft.fft(arr)*rhs
    return np.fft.ifft(conv, n_t).real


def correlation_func(f, g):
    f_p = np.fft.fft(f)
    g_p_conj = np.conjugate(np.fft.fft(g))
    conv = np.fft.ifft(f_p*g_p_conj).real
    return conv/conv.max()


def problem_3():
    n_pts = 100
    shifts = np.linspace(0, 100, 5)[:-1]
    x = np.linspace(-5, 5, n_pts)
    y = gaussian(x)

    fig, axs = plt.subplots(4, figsize=(8, 12), sharex=True)
    fig.suptitle('Correlation between a gaussian and shifted gaussians')
    for i, shift in enumerate(shifts):
        y_shifted = shift_fft(y, shift)
        corr = correlation_func(y, y_shifted)
        axs[i].set_title(f'shift={shift/n_pts}')
        axs[i].plot(y, label='Original Gaussian')
        axs[i].plot(y_shifted, label='Shifted Gaussian')
        axs[i].plot(corr, '--', label='Correlation')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig('Results/q3.png')
    plt.clf()
    plt.close()

    print(
        """
        Problem 3)

        By shifting the Gaussian, The correlation function also shifts,
        expectedly.
        """)