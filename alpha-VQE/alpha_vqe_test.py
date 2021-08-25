from numpy.core.fromnumeric import transpose
from alpha_vqe import Alpha_VQE
import numpy as np
import random
from numpy import pi
from progress.bar import FillingCirclesBar, FillingSquaresBar
import os
import pickle

alpha_values = np.linspace(0, 1, 11)
sample_sizes = [100, 500, 1000]
mRange = 10000

# bar = FillingSquaresBar("Running simulation", max = mRange*len(sample_sizes)*len(alpha_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')

results = []
for ss in sample_sizes:
    print(f"Beginning sample size {ss}")
    save_here = open("alpha-VQE/data/alpha_vqe_ss%s_alpha"%ss, "wb")
    for al in alpha_values:
        print(f"  -Starting simulation on alpha = {al}")
        temp = []
        solutions = 0
        failures = 0
        while solutions < mRange:
            phi = random.uniform(-pi, pi)
            a = Alpha_VQE(phi=phi, accuracy = 0.005, sigma=pi/2, nSamples=ss, alpha = al, max_shots = 10**6)
            err, run, f = a.estimate_phase()

            if f == 1:
                failures += 1
            else:
                temp.append([err, run])
                solutions += 1
                # bar.next()
        results.append(
            [ss, np.median(transpose(temp)[0]), np.median(transpose(temp)[1]), failures / (mRange + failures)]
        )
        print(f"  Ending :: Failure rate {failures}."
            f"\n    Of {mRange} runs, this implies failure rate of" 
            f" {100*failures / (failures + mRange):.2f}%%\n")

    pickle.dump(results, save_here)
    save_here.close()

# bar.finish()


