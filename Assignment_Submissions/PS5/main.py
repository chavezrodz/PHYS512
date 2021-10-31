import sys
import os
from q1 import problem_1
from q2 import problem_2
from q3 import problem_3
from q4 import problem_4
from q5 import problem_5
from q6 import problem_6


os.makedirs('Results', exist_ok=True)

sys.stdout = open('Results/results.txt', 'w+')
problem_1()
problem_2()
problem_3()
problem_4()
problem_5()
problem_6()
sys.stdout.close()
