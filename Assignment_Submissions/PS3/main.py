from q1 import problem_1
from q2 import problem_2
from q3 import problem_3
import sys
import os

os.makedirs('Results', exist_ok=True)
sys.stdout = open('Results/results.txt', 'w+')
problem_1()
problem_2()
problem_3()
sys.stdout.close()
