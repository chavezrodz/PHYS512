import numpy as np
import time
from matplotlib import pyplot as plt


def gauss(x, sigma):
    out = np.exp(
        -np.square((x)/(2*sigma))
        )
    return out


def lorentz(x):
    return 1/(1 + (x)**2)


x = np.linspace(0, 5, 100)
plt.plot(x, np.exp(-x), label="Exponential", linestyle='--')
plt.plot(x, gauss(x, 1), label="Gaussian")
plt.plot(x, lorentz(x), label="Lorentzian")
plt.plot(x, np.power(x+1, -1), label="Power")

plt.legend()
plt.tight_layout()
plt.savefig('Results/q2a.png')
plt.close()

# NB speed of gaussian and power <-1


n = int(1e6)
deviates = np.random.rand(n)
deviates = np.tan(np.pi*(deviates - 0.5))

sample = deviates[np.abs(deviates) < 20]
aa, bb = np.histogram(sample, bins=300)
b_cent = 0.5*(bb[1:]+bb[:-1])
xx = np.linspace(-20, 20, 300)
plt.plot(xx, lorentz(xx), color='red', label='Lorentz Function')
plt.bar(b_cent, aa/aa.max(), 0.15, alpha=0.5, label='Deviates')
plt.tight_layout()
plt.savefig('Results/q2b.png')
plt.close()

deviates = np.abs(deviates)

p = np.exp(-deviates)/lorentz(deviates)
accepted_dev = deviates[np.random.rand(n) < p]

accept_pc = 100*len(accepted_dev)/n
print(f'Accepted deviates percent: {accept_pc}')
aa, bb = np.histogram(accepted_dev, bins=300)
b_cent = 0.5*(bb[1:]+bb[:-1])
pred = np.exp(-b_cent)
pred = pred/pred.max()
aa = aa/aa.max()
plt.plot(b_cent, pred, color='red', label='Exponential')
plt.bar(b_cent, aa, 0.15, label='Lorentzian')
plt.legend()
plt.tight_layout()
plt.savefig('Results/q2c.png')