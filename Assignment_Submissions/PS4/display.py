import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import seaborn as sns

os.makedirs('Results/chain', exist_ok=True)
chain = pd.read_csv('Results/planck_chain.csv')
chain_tau = pd.read_csv('Results/planck_chain_tauprior.csv')

params = ['H0', 'Ohmbh2', 'Ohmch2', 'Tau', 'As', 'ns']


plt.plot(chain['chisq'])
plt.xlabel('Step')
plt.ylabel(r'$\chi^2$')
plt.savefig(os.path.join('Results', 'chain', 'chisq.png'))
plt.clf()


burn = 150
for param in params:
	plt.plot(chain[param].iloc[burn:])
	plt.xlabel('Step')
	plt.ylabel(param)
	plt.savefig(os.path.join('Results', 'chain', param+'.png'))
	plt.clf()

	fft = np.abs(np.fft.rfft(chain[param].iloc[burn:]))
	plt.loglog(fft)
	plt.xlabel('Frequency')
	plt.ylabel('fft_' + param)
	plt.savefig(os.path.join('Results', 'chain', param+'_fft.png'))
	plt.clf()

