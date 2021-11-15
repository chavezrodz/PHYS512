import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import sys
import os
import json
import h5py
from simple_read_ligo import load_event
from analyze_event import analyze_event
import utils as ut

data_folder = 'data/'
results_folder = 'Results'
os.makedirs(results_folder, exist_ok=True)

sys.stdout = open('Results/results.txt', 'w+')

filejson = os.path.join(data_folder, 'BBH_events_v3.json')
events = json.load(open(filejson, 'r'))

event_list = ['GW150914', 'GW151226', 'GW170104', 'LVT151012']


print(
    """
    a)
    We model the noise of the raw data by considering the power spectrum.
    We use the blackman window and use a gaussian filter on the spectrum
    to smoothen out the lines.

    b)
    Using our noise model from a), we first whiten our data then perform
    the matched filtering. Note shift the fft to have the peak at the center

    c)
    we use our previous noise mondel and the matched filter to extract
    the snr

    d)
    we get the analytic snr wihout use of the matched filter

    e)
    we compute a cumulative sum of the power spectrum and normalize it by its
    maximum value. The 50 percentile frequency is then where this cumulative
    sum is 0.5

    f)
    We estimate the time of arrival by fitting a gaussian to the SNR around its
    peak. We get edge of the array errors so cannot fit the gaussian. We take
    the positional uncertainty as the uncertainty in time of arrival.

    We estimate the positional uncertainty by dividing the distance between
    the two detectors by the speed of light
    """
    )


for event in event_list:
    print(f'\t Analyzing event {event}')

    event_dir = os.path.join(results_folder, event)
    os.makedirs(event_dir, exist_ok=True)

    event_data = load_event(event, data_folder, events)

    analyze_event(event_data, event_dir)

sys.stdout.close()
