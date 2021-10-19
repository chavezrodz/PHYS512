import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import seaborn as sns

files = ['chain_tauprior', 'chain']
for file in files:
    os.makedirs(os.path.join('Results', file), exist_ok=True)
    chain = pd.read_csv('Results/planck_'+file+'.csv')

    params = ['H0', 'Ohmbh2', 'Ohmch2', 'Tau', 'As', 'ns']


    plt.plot(chain['chisq'])
    plt.xlabel('Step')
    plt.ylabel(r'$\chi^2$')
    plt.savefig(os.path.join('Results', file, 'chisq.png'))
    plt.clf()


    burn = 150
    for param in params:
        plt.plot(chain[param].iloc[burn:])
        plt.xlabel('Step')
        plt.ylabel(param)
        plt.savefig(os.path.join('Results', file, param+'.png'))
        plt.clf()

        fft = np.abs(np.fft.rfft(chain[param].iloc[burn:]))
        plt.loglog(fft)
        plt.xlabel('Frequency')
        plt.ylabel('fft_' + param)
        plt.savefig(os.path.join('Results', file, param+'_fft.png'))
        plt.clf()

