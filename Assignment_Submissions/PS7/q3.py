import numpy as np
import time
from matplotlib import pyplot as plt

n = int(1e6)
scale = 2/np.e
u, v = np.random.rand(2, n)
v *= scale

ratio = v/u

accepted_dev = ratio[u < np.sqrt(np.exp(-ratio))]

accept_pc = 100*len(accepted_dev)/n
print(accept_pc)

aa, bb = np.histogram(accepted_dev, bins=300)
b_cent = 0.5*(bb[1:]+bb[:-1])
pred = np.exp(-b_cent)
pred = pred/pred.max()
aa = aa/aa.max()
plt.plot(b_cent, pred, color='red', label='Exponential')
plt.bar(b_cent, aa, 0.15, label='Lorentzian')
plt.legend()
plt.savefig('Results/q3.png')
plt.show()
