from avqe import AVQE
import numpy as np
import random
from numpy import pi
from progress.bar import FillingCirclesBar
import os
import pickle
from numpy.core.fromnumeric import transpose


acc = 0.005
alpha_values = np.linspace(0, 1, 21)
max_depth_values = [np.ceil(1/acc**al) for al in alpha_values]


mRange = 500

# bar = FillingCirclesBar("Running simulation", max = mRange*len(max_depth_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')

results = []
for max_d in max_depth_values:
    print(f"Starting simulation on maximum M = {max_d}")
    temp = []
    
    failures = 0
    solutions = 0
    
    theoretical_max = AVQE(phi=0, max_unitaries=max_d).get_max_shots()
    practical_max = np.ceil(1.1 * theoretical_max)

    while solutions < mRange:
        phi = random.uniform(-pi, pi)

        b = AVQE(phi=phi, max_unitaries=max_d, max_shots=practical_max)
        err, run = b.estimate_phase()
        
        if run > practical_max:
            # print(f"Failure; err {err}")
            failures += 1
            continue
        else:
            temp.append([err, run])
            # bar.next()
            solutions += 1
    print(f"  Ending :: Failure rate {failures}."
        f"\n    Of {mRange} runs, this implies failure rate of" 
        f" {100*failures / (failures + mRange):.2f}%%")
    results.append(
        [max_d, np.median(transpose(temp)[0]), np.median(transpose(temp)[1]), failures]
    )
# bar.finish()
# print(results)
f = open("data/avqe_test_acc_%s"%acc, "wb")
pickle.dump(results, f)
