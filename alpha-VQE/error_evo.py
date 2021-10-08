import numpy as np
from numpy import pi, mean, median
from alpha_vqe_rfpe import Alpha_VQE as rfpe
from alpha_vqe import Alpha_VQE as aqpe
from numpy.core.fromnumeric import transpose
import pickle

import sys

from progress.bar import Bar

R = 1000
data = []

bar = Bar('Simulating', max = R, suffix = '%(percent).2f%%')

exp_num = 500
for _ in range(R):
    phi = np.random.uniform(-pi,pi)
    _10k_exp = rfpe(phi=phi, sigma=pi/4, nSamples=400, alpha = 0)
    d = _10k_exp.run_experiment(exp_num)
    data.append(d)
    bar.next()

_10_avgd = [np.median(x) for x in transpose(data)]

bar.finish()
save_here = open("alpha-VQE/data/error_evo", "wb")
pickle.dump(_10_avgd, save_here)


import matplotlib.pyplot as plt

plt.plot(range(exp_num), _10_avgd)
plt.xlabel("Experiment")
plt.ylabel("Median Error")
plt.grid(True)
plt.yscale("log")
plt.ylim(10**-15, 10**0)
plt.legend()
plt.savefig("alpha-VQE/plots/error_evo.pdf")
plt.clf()
