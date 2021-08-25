import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
from numpy import pi
from alpha_vqe import Alpha_VQE as aqpe
from numpy.core.fromnumeric import transpose

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Franklin Gothic Book'
matplotlib.rcParams['axes.linewidth'] = 1.2
colours=['#1d3c34','#00968f','#3ccbda', '#ff7500','#dc4405']

exp_num = 150
ss = [50, 100, 400]

data_exact = []
for _ in range(10000):
    phi = np.random.uniform(-pi,pi)
    exact_sim = aqpe(phi=phi, sigma=pi/2, nSamples=1, alpha = 1, update = 2)
    exact_res = exact_sim.run_experiment(exp_num) 
    data_exact.append(exact_res)
exact_avgd = [np.median(x) for x in transpose(data_exact)]



in1 = open("alpha-VQE/data/alpha_vqe_rfpe_compare", "rb")
aqpe_avgd = pickle.load(in1)

in2 = open("alpha-VQE/data/rfpe_compare_test", "rb")
rfpe_avgd = pickle.load(in2)

for idx, ele in enumerate(rfpe_avgd):
    plt.plot(range(exp_num), ele, label = "k=%s"%ss[idx])
plt.plot(range(exp_num), exact_avgd, '--', color = 'grey', alpha = 0.5 )

plt.xlabel("Experiment")
plt.ylabel("Median Error")
plt.grid(True)
plt.yscale("log")
plt.ylim(10**-15, 10**0)
plt.legend()
plt.savefig("alpha-VQE/plots/rfpe.pdf")
plt.clf()


for idx, ele in enumerate(aqpe_avgd):
    plt.plot(range(exp_num), ele, label = "k=%s"%ss[idx])
plt.plot(range(exp_num), exact_avgd, '--', color = 'grey', alpha = 0.5 )

plt.xlabel("Experiment")
plt.ylabel("Median Error")
plt.grid(True)
plt.yscale("log")
plt.ylim(10**-15, 10**0)
plt.legend()
plt.savefig("alpha-VQE/plots/aqpe.pdf")
