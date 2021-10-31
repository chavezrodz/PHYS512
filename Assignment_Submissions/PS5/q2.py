import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu=0, sigma=1):
    return np.exp(-((x-mu)/(2*sigma))**2)


def correlation_func(f, g):
    f_p = np.fft.fft(f)
    g_p_conj = np.conjugate(np.fft.fft(g))
    conv = np.fft.ifft(f_p*g_p_conj).real
    return conv/conv.max()


def problem_2():
    n_pts = 100
    x = np.linspace(-5, 5, n_pts)
    y = gaussian(x)

    corr = correlation_func(y, y)
    plt.plot(x, y, label='Gaussian')
    plt.plot(x, corr, label='Gaussian Self Correlation')
    plt.legend()
    plt.savefig('Results/q2.png')
    plt.clf()
    plt.close()

    print(
        """
        Problem 2

        Please find the plot in Results/q2.png

        """)
