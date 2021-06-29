from avqe import AVQE
import numpy as np
import random
from numpy import pi
from progress.bar import FillingCirclesBar
import os
import pickle
from numpy.core.fromnumeric import transpose


acc = 0.005
alpha_values = np.linspace(0, 1, 11)
max_depth_values = [np.ceil(1/acc**al) for al in alpha_values]


mRange = 50

bar = FillingCirclesBar("Running simulation", max = mRange*len(max_depth_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')

results = []
for max_d in max_depth_values:
    temp = []
    solutions = 0
    while solutions < mRange:
        phi = random.uniform(-pi, pi)
    
        theoretical_max = AVQE(phi=phi, max_depth=max_d).get_max_shots()
        practical_max = np.ceil(1.01 * theoretical_max)

        b = AVQE(phi=phi, max_depth=max_d, max_shots=practical_max)
        err, run = b.estimate_phase()
        
        if run > practical_max:
            continue
        else:
            temp.append([err, run])
            bar.next()
            solutions += 1
    results.append(
        [max_d, np.median(transpose(temp)[0]), np.median(transpose(temp)[1])]
    )
bar.finish()
print(results)
f = open("data_test_avqe", "wb")
pickle.dump(results, f)
