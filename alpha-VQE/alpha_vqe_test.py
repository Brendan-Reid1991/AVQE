from numpy.core.fromnumeric import transpose
from alpha_vqe import Alpha_VQE
import numpy as np
import random
from numpy import pi
from progress.bar import FillingCirclesBar, FillingSquaresBar
import os
import pickle

alpha_values = np.linspace(0, 1, 11)
sample_sizes = [100, 250, 500, 750, 1000]
mRange = 500

# bar = FillingCirclesBar("Running simulation", max = mRange*len(alpha_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')

# results = []
# for alpha in alpha_values:
#     temp = []
#     for _ in range(mRange):
#         phi = random.uniform(-pi, pi)
#         a = Alpha_VQE(phi=phi, nSamples=100, alpha = alpha, rescaled=1)
#         max_shots = a.get_max_shots()
#         err, run = a.estimate_phase()
#         temp.append([err, run, max_shots])
#         bar.next()
#     results.append(
#         [alpha, np.median(transpose(temp)[0]), np.median(transpose(temp)[1])]
#     )
# bar.finish()

# f = open("data_test", "wb")
# pickle.dump(results, f)

bar = FillingSquaresBar("Running simulation", max = mRange*len(sample_sizes)*len(alpha_values), suffix = '%(percent).2f%% [%(elapsed_td)s]')

results = []
for ss in sample_sizes:
    save_here = open("alpha-VQE/data/alpha_vqe_ss%s_alpha"%ss, "wb")
    for al in alpha_values:
        temp = []
        for _ in range(mRange):
            phi = random.uniform(-pi, pi)
            a = Alpha_VQE(phi=phi, nSamples=ss, alpha = al)
            max_shots = a.get_max_shots()
            err, run = a.estimate_phase()
            temp.append([err, run])
            bar.next()
        results.append(
            [ss, np.median(transpose(temp)[0]), np.median(transpose(temp)[1])]
        )
    pickle.dump(results, save_here)
    save_here.close()

bar.finish()


