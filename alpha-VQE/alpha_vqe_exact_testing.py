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

# bar = FillingSquaresBar("Running exact simulation", max = mRange*len(alpha_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')

exact_results = []
for al in alpha_values:
    print(f"Starting simulation on alpha = {al}")
    temp = []
    failures = 0
    solutions = 0

    while solutions < mRange:
        phi = random.uniform(-pi, pi)
        a = Alpha_VQE(phi=phi, nSamples=1, sigma = pi/4, alpha = al, update = 2, max_shots=2*10**5)
        err, run, failed = a.estimate_phase()
        if failed == 1:
            failures += 1
        else:
            temp.append([err, run, failed])
            solutions += 1
            # bar.next()
    print(f"  Ending :: Failure rate {failures}."
        f"\n    Of {mRange} runs, this implies failure rate of" 
        f" {100*failures / (failures + mRange):.2f}%%")
    
    exact_results.append([
        np.median(transpose(temp)[0]), np.median(transpose(temp)[1]), failures / (mRange + failures)
    ])    
    print(
        f"  Average number of runs: {np.mean(transpose(temp)[1])}"
    )
# bar.finish()
import matplotlib.pyplot as plt

# plt.plot(exact_results[0])
# plt.show()
g = open("alpha-VQE/data/alpha_exact_alpha", "wb")
pickle.dump(exact_results, g)
# print(exact_results)