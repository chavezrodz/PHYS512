import sys
import os
from q2a import problem2a
from q2b import problem2b
from q2c import problem2c


os.makedirs('Results', exist_ok=True)

sys.stdout = open('Results/results.txt', 'w+')


n = 128
side = 16
edge = 24
problem2a(n)
problem2b(n, side)
problem2c(n, side, edge)
sys.stdout.close()
