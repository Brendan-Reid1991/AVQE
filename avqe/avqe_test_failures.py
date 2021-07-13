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
sigmas = [pi/2, pi/4, pi/8, pi/16]

mRange = 1000

# bar = FillingCirclesBar("Running simulation", max = mRange*len(max_depth_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')

suffixes = [2,4,8,16]
for idx, init_sig in enumerate(sigmas):
    results = []
    write_here = "avqe_failure_sig_pi-%s"%suffixes[idx]
    for max_d in max_depth_values:
        print(f"Starting simulation on maximum M = {max_d}")
        temp = []
        
        failures = 0
        solutions = 0
        
        while solutions < mRange:
            phi = random.uniform(-pi, pi)

            b = AVQE(phi=phi, max_unitaries=max_d)
            err, run, f = b.estimate_phase()
            if f == 1:
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
    f = open("avqe/data/"+write_here, "wb")
    pickle.dump(results, f)
# bar.finish()
# print(results)

