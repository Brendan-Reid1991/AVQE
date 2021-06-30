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

# bar = FillingSquaresBar("Running simulation", max = mRange*len(sample_sizes), suffix = '%(percent).2f%% [%(elapsed_td)s]')

# results = []
# for ss in sample_sizes:
#     temp = []
#     for _ in range(mRange):
#         phi = random.uniform(-pi, pi)
#         a = Alpha_VQE(phi=phi, nSamples=ss, alpha = 0.5, rescaled=1)
#         max_shots = a.get_max_shots()
#         err, run = a.estimate_phase()
#         temp.append([err, run])
#         bar.next()
#     results.append(
#         [ss, np.median(transpose(temp)[0]), np.median(transpose(temp)[1])]
#     )
# bar.finish()

# f = open("alpha_vqe_sample_test", "wb")
# pickle.dump(results, f)

bar = FillingSquaresBar("Running exact simulation", max = mRange, suffix = '%(percent).2f%% [%(elapsed_td)s]')
exact_results = []
for _ in range(mRange):
    phi = random.uniform(-pi, pi)
    a = Alpha_VQE(phi=phi, nSamples=1, alpha = 0.5, exact = 1)
    err, run = a.estimate_phase()
    exact_results.append([err, run])
    bar.next()
bar.finish()
print(exact_results)
g = open("alpha_vqe_exact_update", "wb")
pickle.dump(exact_results, g)