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
max_depth_values = [np.round(1/acc**al) for al in alpha_values]

sigma_values = [pi/16, pi/8, pi/4, pi/2]
label = ["pi-16", "pi-8", "pi-4", "pi-2"]

mRange = 2500

# bar = FillingCirclesBar("Running simulation", max = mRange*len(max_depth_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')

collated_results = []
for idx,sigma in enumerate(sigma_values):
    print(f"Starting standard deviation {sigma:.4f}")
    results = []
    for max_d in max_depth_values:
        print(f"  Starting simulation on maximum M = {max_d}")
        temp = []
        
        failures = 0
        solutions = 0
        bar = FillingCirclesBar("  Running simulation", max = mRange, suffix = '%(percent).2f%% [%(elapsed_td)s]')


        while solutions < mRange:
            phi = random.uniform(-pi, pi)

            b = AVQE(phi=phi, max_unitaries=max_d, accuracy = acc, sigma = sigma, max_shots = 2*10**5, state = 1)
            err, run, f = b.estimate_phase()
            
            if f == 1:
                # print(f"{solutions} |  Failure; err {err}")
                failures += 1
                continue
            else:
                temp.append([err, run])
                bar.next()
                # print(f"{solutions} | success")
                solutions += 1
        print(f"  Ending :: Failure rate {failures}."
            f"\n    Of {mRange} runs, this implies failure rate of" 
            f" {100*failures / (failures + mRange):.2f}%%")
        results.append(
            [max_d, np.median(transpose(temp)[0]), np.median(transpose(temp)[1]), failures / (mRange + failures)]
        )
        bar.finish()
    # collated_results.append(results)

    f = open("avqe/data/avqe_sup_test_sigma_%s"%label[idx], "wb")
    pickle.dump(results, f)
