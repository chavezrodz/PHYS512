import numpy as np
import matplotlib.pyplot as plt


def analytic_sine_dft(k, N):
    kvec = np.arange(N)

    f1 = (1 - np.exp(-2j*np.pi*(kvec-k)))/((1 - np.exp(-2j*np.pi*(kvec-k)/N)))
    f2 = (1 - np.exp(-2j*np.pi*(kvec+k)))/((1 - np.exp(-2j*np.pi*(kvec+k)/N)))

    return np.abs((f1 - f2)/2j)


def window(x):
    return 0.5*(1 - np.cos(2*np.pi*x))


def problem_5():

    k = np.pi
    N = 100
    x = np.arange(N)

    y = np.sin(2*np.pi*x*k/N)
    analytic = analytic_sine_dft(k, N)
    FFT = np.abs(np.fft.fft(y))

    err_analytic = FFT-analytic

    # c
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Fourier Transform of a sine wave')
    axs[0].plot(FFT, '-', label='Numerical FFT')
    axs[0].plot(analytic, '--', label='Analytical FFT')
    axs[0].legend()

    axs[1].set_ylabel('Residuals')
    axs[1].scatter(x, err_analytic, marker='.')

    plt.savefig('Results/q5c.png')
    plt.clf()
    plt.close()

    # d
    wind = window(x/N)
    FFT_windowed = np.abs(np.fft.fft(y*wind))

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Comparing FFT with and without window')

    axs[0].plot(y, '-', label='Sine Function')
    axs[0].plot(wind, '--', label='Window Function')
    axs[0].legend()

    axs[1].set_ylabel('FFT')
    axs[1].plot(FFT, '-', label='Unwindowed FFT')
    axs[1].plot(FFT_windowed, '--', label='Windowed FFT')
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('Results/q5d.png')
    plt.clf()
    plt.close()

    # e

    num_window_fft = np.fft.fft(window(x/N)).real
    window_fft = np.zeros(N)
    window_fft[0] = N/2
    window_fft[[1, -1]] = -N/4

    err_wind = num_window_fft - window_fft

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Numerical Window FFT vs suggested Window FFT')

    axs[0].plot(num_window_fft, '-', label='Numerical FFT')
    axs[0].plot(window_fft, '--', label='Suggested FFT')
    axs[0].legend()

    axs[1].set_ylabel('Residuals')
    axs[1].set_xlabel('K')
    axs[1].scatter(x, err_wind, marker='.')

    plt.tight_layout()
    plt.savefig('Results/q5e1.png')
    plt.clf()
    plt.close()

    def windowed_fft(fft):
        N = len(fft)
        windowed = []
        for n in range(N):
            windowed.append(fft[n]/2 - (fft[(n + 1)%N] + fft[(n - 1)%N])/4)
        return np.abs(windowed)

    FFT_windowed_array = windowed_fft(FFT)
    err_wind_array = FFT_windowed - FFT_windowed_array

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Numerical Window FFT vs FFT windowed using array')
    axs[0].plot(FFT_windowed, '-', label='Windowed FFT')
    axs[0].plot(FFT_windowed_array, '--', label='FFT windowed using array')
    axs[0].legend()

    axs[1].set_ylabel('Residuals')
    axs[1].set_xlabel('K')
    axs[1].scatter(x, err_wind_array, marker='.')
    plt.tight_layout()
    plt.savefig('Results/q5e2.png')
    plt.clf()
    plt.close()

    print(
        f"""
        Problem 5)

        Please find a) & b) in 5ab.pdf

        c)
        The Numerical FFT of the sine function agrees with our analytical
        FFT with average error {err_analytic.mean():.3e}

        We have the two expected peaks at +/- k, however, the width of the delta functions
        is rather broad

        d)
        We define the window function, and plot the result in 5d.png.
        We note that the width of the delta function function decreases with windowing,
        reducing leakage.

        e)
        We show numerically that the FFT of the window function is [N/2, -N/4, 0, ..., -N/4]
        by comparing the numerical fft and the suggested array. We obtain the results to agree
        to machine precision, with average error {err_wind.mean():.3e}

        We show the demonstration of the windowed fft using the previous part in the attached
        image, then compare 
            -the fft of a windowed sine wave
            -the fft of an unwindowed sine wave, windowed using the suggested array
        Both methods agree to an average error {err_wind_array.mean():.3e}


        """)
