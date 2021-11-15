import numpy as np
import sys
import os
import json
import h5py
from plotting import *
from astropy import units as u
import utils as ut

titlesize = 20
labelsize = 14
legsize = 14


def analyze_event(event_data, event_dir):
    time, fs, strains, templs = event_data
    strain_H1, strain_L1 = strains
    templ_H1, templ_L1 = templs
    toff = time.min()
    offset_time = time - toff

    freqs = np.fft.rfftfreq(strain_H1.size, 1.0/fs)

    # Visualizing raw data
    plot_raw(strain_H1, strain_L1, templ_H1, templ_L1, offset_time,
             titlesize, labelsize, legsize, event_dir)

    # a) Noise Model
    window = np.blackman(strain_H1.size)

    # Power spectrum for each detector
    powers_H1 = ut.power_spectrum(strain_H1, window=window)
    powers_L1 = ut.power_spectrum(strain_L1, window=window)

    plot_noise_model(freqs, powers_H1, powers_L1,
                     titlesize, labelsize, legsize, event_dir)

    # b) Matched Filter

    mf_H1 = ut.matched_filter(strain_H1, templ_H1, powers_H1, window=window)
    mf_L1 = ut.matched_filter(strain_L1, templ_L1, powers_L1, window=window)

    plot_filter(mf_H1, mf_L1, offset_time,
                titlesize, labelsize, legsize, event_dir)

    # c) Signal to Noise Ratio

    snr_H1 = ut.get_snr(mf_H1, templ_H1, powers_H1, window=window)
    snr_L1 = ut.get_snr(mf_L1, templ_L1, powers_L1, window=window)
    snr_tot = np.sqrt(snr_H1**2 + snr_L1**2)

    print(f'Max SNR H1: {np.max(snr_H1):.4f}')
    print(f'Max SNR L1: {np.max(snr_L1):.4f}')
    print(f'Max SNR (total): {np.max(snr_tot):.4f}')

    plot_SNR(snr_H1, snr_L1, snr_tot, offset_time,
             titlesize, labelsize, legsize, event_dir)

    # d) Analytic Signal to noise ratio

    esnr_H1 = ut.expected_snr(templ_H1, powers_H1, window=window)
    esnr_L1 = ut.expected_snr(templ_L1, powers_L1, window=window)
    esnr_tot = np.sqrt(esnr_H1**2 + esnr_L1**2)

    print(f'Max Analytic SNR H1: {np.max(esnr_H1):.4f}')
    print(f'Max Analytic SNR L1: {np.max(esnr_L1):.4f}')
    print(f'Max Analytic SNR (total): {np.max(esnr_tot):.4f}')

    plot_analytic_SNR(esnr_H1, esnr_L1, esnr_tot, offset_time,
                      titlesize, labelsize, legsize, event_dir)

    # e) Half Power Frequency

    hf_H1 = ut.half_pc_freq(freqs, templ_H1, powers_H1, window=window)
    hf_L1 = ut.half_pc_freq(freqs, templ_L1, powers_L1, window=window)

    print(f'Half Power Frequency for H1: {hf_H1} Hz')
    print(f'Half Power Frequency for L1: {hf_L1} Hz')

    # f) Time on Arrival

    imax_H1 = np.argmax(snr_H1)
    imax_L1 = np.argmax(snr_L1)

    # find snr peak time
    # This fails because of edge cases

    # nedge = 10
    # Indexes near arrival
    # idx_H1 = np.arange(imax_H1-nedge, imax_H1+nedge)
    # idx_L1 = np.arange(imax_H1-nedge, imax_H1+nedge)

    # get time and uncertainty for each detector
    # ta_H1, eta_H1 = ut.toa(time, snr_H1, nside=nedge)
    # ta_L1, eta_L1 = ut.toa(time, snr_L1, nside=nedge)

    ta_H1 = time[imax_H1]
    ta_L1 = time[imax_L1]

    print(f'H1 time of arrival: {ta_H1}')
    print(f'L1 time of arrival: {ta_L1}')
    delta_t = np.abs(ta_H1 - ta_L1)

    pos_err = 3000e3 / 3e8
    delta_err = 2 * pos_err
    print(f'Difference in time of arrival: {delta_t:.4e} ± {delta_err}\n')