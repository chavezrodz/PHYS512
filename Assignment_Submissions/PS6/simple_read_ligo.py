import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob


def read_template(filename):
    dataFile = h5py.File(filename, 'r')
    template = dataFile['template']
    th = template[0]
    tl = template[1]
    return th, tl


def read_file(filename):
    dataFile = h5py.File(filename, 'r')
    dqInfo = dataFile['quality']['simple']
    qmask = dqInfo['DQmask'][...]

    meta = dataFile['meta']
    gpsStart = meta['GPSstart'][()]
    utc = meta['UTCstart'][()]
    duration = meta['Duration'][()]
    strain = dataFile['strain']['Strain'][()]

    dataFile.close()

    return strain, gpsStart, duration


def load_event(eventname, datadir, events):
    # get event info from json file
    event = events[eventname]
    fn_H1 = event['fn_H1']
    fn_L1 = event['fn_L1']
    fn_template = event['fn_template']
    fs = event['fs']

    strain_H1, start, duration = read_file(datadir+fn_H1)
    strain_L1, _, _ = read_file(datadir+fn_L1)
    strains = (strain_H1, strain_L1)

    dt = duration/len(strain_H1)
    time = np.arange(start, start + duration, dt)

    templ_H1, templ_L1 = read_template(datadir+fn_template)
    templs = (templ_H1, templ_L1)

    return time, fs, strains, templs
