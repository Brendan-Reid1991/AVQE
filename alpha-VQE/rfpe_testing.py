import numpy as np
from numpy import pi, mean, median
from alpha_vqe_rfpe import Alpha_VQE as rfpe
from alpha_vqe import Alpha_VQE as aqpe
from numpy.core.fromnumeric import transpose
import pickle

import sys

data_rfpe = []
data_aqpe = []

exp_num = 150
for _ in range(10000):
    phi = np.random.uniform(-pi,pi)
    ss50 = rfpe(phi=phi, sigma=pi/2, nSamples=50, alpha = 0.8, update = 1)
    ss100 = rfpe(phi=phi, sigma=pi/2, nSamples=100, alpha = 0.8, update = 1)
    ss400 = rfpe(phi=phi, sigma=pi/2, nSamples=400, alpha = 0.8, update = 1)
    b = rfpe(phi=phi, sigma=pi/2, nSamples=1, alpha = 0.8, update = 2)
    rf50 = ss50.run_experiment(exp_num)    
    rf100 = ss100.run_experiment(exp_num)    
    rf400 = ss400.run_experiment(exp_num)    
    
    aq = b.run_experiment(exp_num)  
    # sys.stdout.write("%s"%_)
    # sys.stdout.flush()

    data_rfpe.append([rf50, rf100, rf400])
    data_aqpe.append(aq)

exact_avgd = [median(x) for x in transpose(data_aqpe)]

rf50 = [data_rfpe[i][0] for i in range(len(data_rfpe))]
rf100 = [data_rfpe[i][1] for i in range(len(data_rfpe))]
rf400 = [data_rfpe[i][2] for i in range(len(data_rfpe))]
ss = [50, 100, 400]
rfpe_avgd = []
for l in [rf50, rf100, rf400]:
    rfpe_avgd.append([np.median(x) for x in transpose(l)])

save_here = open("alpha-VQE/data/rfpe_compare_test_alpha_pt8", "wb")
pickle.dump(rfpe_avgd, save_here)


import matplotlib.pyplot as plt
for idx, ele in enumerate(rfpe_avgd):
    plt.plot(range(exp_num), ele, label = "m=%s"%ss[idx])
plt.plot(range(exp_num), exact_avgd, '--', color = 'grey', label = 'Exact')
plt.xlabel("Experiment")
plt.ylabel("Median Error")
plt.grid(True)
plt.yscale("log")
plt.ylim(10**-15, 10**0)
plt.legend()
plt.savefig("alpha-VQE/plots/rfpe_test_pt8.pdf")
plt.clf()
