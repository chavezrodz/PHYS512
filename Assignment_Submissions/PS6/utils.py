import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


def power_spectrum(y, window=None, sigma=1):
    window = window if window is not None else np.ones_like(y)

    # normalizing factor from windowing
    norm = np.sqrt(np.square(window).mean())

    fft = np.fft.rfft(y * window) / norm
    spect = np.abs(fft)**2
    # Smoothen out with Gaussian filter
    spect = gaussian_filter(spect, sigma)

    return spect


def matched_filter(y, template, noise, window=None):
    window = window if window is not None else np.ones_like(y)
    norm = np.sqrt(np.square(window).mean())

    strain_ft = np.fft.rfft(y * window) / (np.sqrt(noise)*norm)
    templ_ft = np.fft.rfft(template * window) / (np.sqrt(noise)*norm)

    mf = np.fft.irfft(strain_ft*np.conj(templ_ft))
    return mf


def get_snr(mf, template, noise, window=None):
    window = window if window is not None else np.ones_like(mf)
    norm = np.sqrt(np.square(window).mean())

    template_ft = np.fft.rfft(template * window) / norm

    snr_rt = np.abs(template_ft)/np.sqrt(noise)
    snr = np.abs(mf * np.fft.irfft(snr_rt))
    return snr


def expected_snr(template, noise, window=None):
    window = window if window is not None else np.ones_like(template)

    norm = np.sqrt(np.mean(window**2))
    template_ft = np.fft.rfft(template * window) / norm

    snr = np.abs(np.fft.irfft(template_ft / np.sqrt(noise)))

    return snr


def half_pc_freq(freqs, template, noise, window):
    window = window if window is not None else np.ones_like(template)

    norm = np.sqrt(np.mean(window**2))
    template_ft = np.fft.rfft(template * window) / norm

    cumulative_power = np.cumsum(np.abs(template_ft**2 / noise))
    cumulative_power = cumulative_power/cumulative_power.max()

    idx = np.argmin(np.abs(cumulative_power - 0.5))

    return freqs[idx]


def gauss(x, a, mu, sigma):
    out = a * np.exp(
        -np.square((x-mu)/(2*sigma))
        )
    return out


def toa(time, snr, nside):
    # nside  (int): number of points to keep on each side of the max
    # Did not end up using
    snr = np.fft.fftshift(snr)
    a = np.max(snr)
    imax = np.argmax(snr)
    est_time = time[imax]
    s0 = 1
    eta, _ = curve_fit(
        lambda x, s: gauss(x, a, est_time, s),
        time[imax-nside:imax+nside],
        snr[imax-nside:imax+nside],
        p0=[s0])

    return est_time, eta.item()