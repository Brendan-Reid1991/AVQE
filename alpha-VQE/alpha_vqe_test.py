from numpy.core.fromnumeric import transpose
from alpha_vqe import Alpha_VQE
import numpy as np
import random
from numpy import pi
from progress.bar import FillingCirclesBar, FillingSquaresBar
import os
import pickle


# errs = []
# runs = []
# for _ in range(10):
#     a = Alpha_VQE(phi=random.uniform(-pi, pi), accuracy = 0.005, sigma=pi/4, nSamples=1000, alpha = 0, max_shots = 10**8)
#     err, run, f, sig = a.estimate_phase()
#     print(run)
#     errs.append(err)
#     runs.append(run)

# print(np.median(errs))
# print(np.mean(errs))

# print(np.median(runs))
# exit()

alpha_values = np.linspace(0, 1, 11)
sample_sizes = [100, 500, 1000]
sample_sizes = [1000]
mRange = 1000

# bar = FillingSquaresBar("Running simulation", max = mRange*len(sample_sizes)*len(alpha_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')


for ss in sample_sizes:
    results = []
    save_here = open("alpha-VQE/data/alpha_vqe_ss%s"%ss, "wb")
    bar = FillingSquaresBar(f"Beginning sample size {ss}", max = mRange*len(alpha_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')
    for al in alpha_values:
        # print(f"  -Starting simulation on alpha = {al}")
        temp = []
        solutions = 0
        failures = 0
        while solutions < mRange:
            phi = random.uniform(-pi, pi)
            a = Alpha_VQE(phi=phi, accuracy = 0.005, sigma=pi/4, nSamples=ss, alpha = al, max_shots = 10**5)
            err, run, f = a.estimate_phase()

            if f == 1:
                failures += 1
            else:
                temp.append([err, run])
                solutions += 1
                bar.next()
        results.append(
            [ss, np.median(transpose(temp)[0]), np.median(transpose(temp)[1]), failures / (mRange + failures)]
        )
    bar.finish()

    pickle.dump(results, save_here)
    save_here.close()

# bar.finish()


