from numpy.core.fromnumeric import transpose
from alpha_vqe import Alpha_VQE
import numpy as np
import random
from numpy import pi
from progress.bar import FillingCirclesBar, FillingSquaresBar
import os
import pickle

alpha_values = np.linspace(0, 1, 21)
sample_sizes = [100, 250, 500, 750, 1000]
mRange = 500

bar = FillingSquaresBar("Running exact simulation", max = mRange*len(alpha_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')

exact_results = []
for al in alpha_values:
    temp = []
    for _ in range(mRange):
        phi = random.uniform(-pi, pi)
        a = Alpha_VQE(phi=phi, nSamples=1, alpha = al, update = 2, max_shots=2*10**6)
        err, run = a.estimate_phase()
        temp.append([err, run])
        bar.next()
    exact_results.append([
        np.median(transpose(temp)[0]), np.median(transpose(temp)[1])
    ])    
bar.finish()

g = open("alpha-VQE/data/alpha_exact_alpha", "wb")
pickle.dump(exact_results, g)