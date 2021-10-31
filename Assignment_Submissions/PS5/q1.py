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


def problem_1():
    n_pts = 100
    x = np.linspace(-5, 5, n_pts+1)
    y = gaussian(x)

    shift = n_pts//4

    yy = shift_fft(y, shift)
    y_mean = x[np.argmax(y)]
    yy_mean = x[np.argmax(yy)]
    vlines = [y_mean, yy_mean]

    print(
        f"""
        Problem 1

        Shifting Gaussian by {shift} points,
        which translates to dx = {shift/np.ptp(x)-x.mean():.2f}.
        mean of gaussian is now at {yy_mean}
        """)
    colors = ['blue', 'orange']
    linestyles = ['-', '--']

    plt.plot(x, y, label='original',
             linestyle=linestyles[0], color=colors[0])

    plt.plot(x, yy, label='shifted',
             linestyle=linestyles[1], color=colors[1])
    plt.ylim(0, 1)
    plt.vlines(vlines, 0, 1, linestyle=linestyles, colors=colors)
    plt.legend()
    plt.savefig('Results/gauss_shift.png')
    plt.clf()
    plt.close()
