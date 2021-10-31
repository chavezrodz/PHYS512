import numpy as np
import matplotlib.pyplot as plt


def conv(f, g):
    fp = np.fft.fft(f)
    gp = np.fft.fft(g)
    conved = np.fft.ifft(fp*gp).real
    return conved/conved.max()


def conv_safe(f, g, padding=0):
    n_f = len(f)
    n_g = len(g)
    print(f'Input arrays of length {n_f} & {n_g}')
    n_min = min(n_f, n_g)
    n_max = max(n_f, n_g)

    if n_f != n_g:
        print('Different input arrays, padding one')
        f = np.pad(f, [0, n_max - n_f])
        g = np.pad(g, [0, n_max - n_g])
        print(f'Arrays now of lengths {len(f)} & {len(g)}')

    if padding != 0:
        print(f'Padding both arrays by {padding} on the ends')
        f = np.pad(f, [padding, padding])
        g = np.pad(g, [padding, padding])

    conved = conv(f, g)

    if n_f == n_g:
        out = conved
        if padding != 0:
            out = conved[padding:-padding]
    else:
        out = conved[:n_min]
        if padding != 0:
            out = conved[padding:n_min+padding]
    print('')
    return out


def problem_4():

    print(
        """
        Problem 4)

        If one input array is shorter than the other, we pad it 
        with zeros so that both arrays are the same length. The
        output we take is the size of the shorter array.

        """)


    padding = 3

    n_pts_1 = 100
    x_min = -12
    x_max = 12
    x1 = np.linspace(x_min, x_max, n_pts_1)

    f = np.cos(x1)
    g1 = 0.2*x1

    conved_1 = conv_safe(f, g1)
    conved_pad_1 = conv_safe(f, g1, padding=padding)

    n_pts_2 = 150
    x_range = np.ptp(x1)
    x2_max = x_min + n_pts_2/n_pts_1 * x_range
    x2 = np.linspace(x_min, x2_max, n_pts_2)
    g2 = 0.2*x2

    conved_2 = conv_safe(f, g2)
    conved_pad_2 = conv_safe(f, g2, padding=padding)

    fig, axs = plt.subplots(2, 2, sharey=True)

    fig.suptitle('Comparing Padding and different input lengths on '
                 'convolutions')
    axs[0, 0].set_ylabel('Same length arrays')
    axs[0, 0].plot(x1, f, label='Original f')
    axs[0, 0].plot(x1, g1, label='Original g')
    axs[0, 0].legend()

    axs[0, 1].plot(x1, conved_1, label='Unpadded')
    axs[0, 1].plot(x1, conved_pad_1, label='Padded')
    axs[0, 1].legend()

    axs[1, 0].set_ylabel('Different length arrays')
    axs[1, 0].plot(x1, f, label='Original f')
    axs[1, 0].plot(x2, g2, label='Original g')
    axs[1, 0].legend()

    axs[1, 1].plot(x1, conved_2, label='Unpadded')
    axs[1, 1].plot(x1, conved_pad_2, label='Padded')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('Results/q4.png')
    plt.clf()
    plt.close()
