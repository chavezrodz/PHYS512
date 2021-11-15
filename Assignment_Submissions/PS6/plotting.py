import numpy as np
import matplotlib.pyplot as plt
import os


def plot_raw(strain_H1, strain_L1, templ_H1, templ_L1, offset_time,
             titlesize, labelsize, legsize, event_dir):

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

    ax[0].plot(offset_time, strain_H1*1e19,
               linewidth=0.5, color='b', label='H1 Data')
    ax[0].plot(offset_time, strain_L1*1e19,
               linewidth=0.5, color='r', label='L1 Data')

    ax[0].set_ylabel(r'Strain ($10^{19}$)', fontsize=labelsize)
    ax[0].legend(fontsize=legsize)
    ax[0].set_title('LIGO Data', fontsize=titlesize)

    ax[1].plot(offset_time, templ_H1*1e19, linewidth=0.5, color='b',
               label='H1 Template')
    ax[1].plot(offset_time, templ_L1*1e19, linewidth=0.5, color='r',
               label='L1 Template')
    ax[1].set_ylabel(r'Strain ($10^{19}$)', fontsize=labelsize)
    ax[1].set_xlabel('Time - offset (s)')
    ax[1].legend(fontsize=legsize)
    plt.savefig(os.path.join(event_dir, 'raw_data.png'))
    plt.clf()
    plt.close()


def plot_noise_model(freqs, powers_H1, powers_L1,
                     titlesize, labelsize, legsize, event_dir):
    plt.loglog(freqs, np.sqrt(powers_H1), 'b', label='H1')
    plt.loglog(freqs, np.sqrt(powers_L1), 'r', label='L1')
    plt.xlim(20, 2000)
    plt.ylabel(r'ASD (strain/$\sqrt{ \t extrm{Hz}}$)', fontsize=labelsize)
    plt.xlabel(r'Frequency (Hz)', fontsize=labelsize)
    plt.title(
        r'Log-log plot of the Amplitude Spectrums',
        fontsize=titlesize)
    plt.legend(fontsize=legsize)
    plt.savefig(os.path.join(event_dir, 'Noise_Model.png'))
    plt.clf()
    plt.close()


def plot_filter(mf_H1, mf_L1, offset_time,
                titlesize, labelsize, legsize, event_dir):

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

    ax[0].plot(offset_time, np.fft.fftshift(mf_H1),
               linewidth=0.5, color='b', label='H1 Output')
    ax[0].set_ylabel(r'Filter Output', fontsize=labelsize)
    ax[0].legend(loc=1, fontsize=legsize)
    ax[0].set_title('Matched Filtering Outputs in Time Domain',
                    fontsize=titlesize)

    ax[1].plot(offset_time, np.fft.fftshift(mf_L1), 
               linewidth=0.5, color='r', label='L1 Output')
    ax[1].set_ylabel(r'Filter Output', fontsize=labelsize)
    ax[1].set_xlabel('Time - offset (s)', fontsize=labelsize)
    ax[1].legend(fontsize=legsize)
    plt.savefig(os.path.join(event_dir, 'Match_filter.png'))
    plt.clf()
    plt.close()


def plot_SNR(snr_H1, snr_L1, snr_tot, offset_time,
             titlesize, labelsize, legsize, event_dir):
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 9))

    ax[0].plot(offset_time, np.fft.fftshift(snr_H1), linewidth=0.5, color='b',
               label='H1 SNR')
    ax[0].set_ylabel(r'SNR', fontsize=labelsize)
    ax[0].legend(loc=1, fontsize=legsize)
    ax[0].set_title('Signal to Noise Ratio (SNR) in Time Domain',
                    fontsize=titlesize)
    ax[1].plot(offset_time, np.fft.fftshift(snr_L1), linewidth=0.5, color='r',
               label='L1 SNR')
    ax[1].set_ylabel(r'SNR', fontsize=labelsize)
    ax[1].legend(loc=1, fontsize=legsize)

    ax[2].plot(offset_time, np.fft.fftshift(snr_tot), linewidth=0.5, color='g',
               label='Combined SNR')
    ax[2].set_ylabel(r'SNR', fontsize=labelsize)
    ax[2].set_xlabel('Time - offset (s)', fontsize=labelsize)
    ax[2].legend(fontsize=legsize)

    plt.savefig(os.path.join(event_dir, 'SNR.png'))
    plt.clf()
    plt.close()


def plot_analytic_SNR(esnr_H1, esnr_L1, esnr_tot, offset_time, 
                      titlesize, labelsize, legsize, event_dir):

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 9))

    ax[0].plot(offset_time, esnr_H1, linewidth=0.5, color='b',
               label='H1 SNR')
    ax[0].set_ylabel(r'SNR', fontsize=labelsize)
    ax[0].legend(loc=1, fontsize=legsize)
    ax[0].set_title('Analytic Expected SNR in Time Domain', fontsize=titlesize)

    ax[1].plot(offset_time, esnr_L1,
               linewidth=0.5, color='r', label='L1 SNR')
    ax[1].set_ylabel(r'SNR', fontsize=labelsize)
    ax[1].legend(loc=1, fontsize=legsize)

    ax[2].plot(offset_time, esnr_tot, 
               linewidth=0.5, color='g', label='Combined SNR')
    ax[2].set_ylabel(r'SNR', fontsize=labelsize)
    ax[2].set_xlabel('Time - offset (s)', fontsize=labelsize)
    ax[2].legend(fontsize=legsize)

    plt.savefig(os.path.join(event_dir, 'Expected_SNR.png'))
    plt.clf()
    plt.close()
