import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
from numpy import pi
from alpha_vqe import Alpha_VQE as aqpe
from alpha_vqe_rfpe import Alpha_VQE as rfpe
from numpy.core.fromnumeric import transpose

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Franklin Gothic Book'
matplotlib.rcParams['axes.linewidth'] = 1.2
colours=['#1d3c34','#00968f','#3ccbda', '#ff7500','#dc4405']

tick_size = 15
label_size = 20
legend_title_size = 15
legend_size = 15

exp_num = 150
ss = [50, 100, 400]

aqpe_exact = []
rfpe_exact = []
for _ in range(10000):
    phi = np.random.uniform(-pi,pi)
    exact_aqpe = aqpe(phi=phi, sigma=pi/2, nSamples=1, alpha = 0.8, update = 2)
    exact_res = exact_aqpe.run_experiment(exp_num) 
    aqpe_exact.append(exact_res)

    exact_rfpe = rfpe(phi=phi, sigma=pi/2, nSamples=1, alpha = 0.8, update = 2)
    exact_res = exact_rfpe.run_experiment(exp_num) 
    rfpe_exact.append(exact_res)
aqpe_exact_avgd = [np.median(x) for x in transpose(aqpe_exact)]
rfpe_exact_avgd = [np.median(x) for x in transpose(rfpe_exact)]



in1 = open("alpha-VQE/data/alpha_vqe_rfpe_compare_pt8", "rb")
aqpe_avgd = pickle.load(in1)

in2 = open("alpha-VQE/data/rfpe_compare_test_pt8", "rb")
rfpe_avgd = pickle.load(in2)

for idx, ele in enumerate(rfpe_avgd):
    plt.plot(range(exp_num), ele, label = "k=%s"%ss[idx])
plt.plot(range(exp_num), rfpe_exact_avgd, '--', color = 'grey', alpha = 0.5, label = 'Exact' )

plt.xlabel("Experiment",fontsize = label_size)
plt.ylabel("Median Error",fontsize = label_size)
plt.grid(True)
plt.yscale("log")
plt.ylim(10**-15, 10**0)
plt.xlim(1, 150)
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)
plt.legend(fontsize = legend_size)
plt.title('RFPE', fontsize = 20)
plt.savefig("alpha-VQE/plots/rfpe_pt8.pdf", bbox_inches = 'tight')
plt.clf()


for idx, ele in enumerate(aqpe_avgd):
    plt.plot(range(exp_num), ele, label = "k=%s"%ss[idx])
plt.plot(range(exp_num), aqpe_exact_avgd, '--', color = 'grey', alpha = 0.5, label = 'Exact' )

plt.xlabel("Experiment",fontsize = label_size)
plt.ylabel("Median Error",fontsize = label_size)
plt.grid(True)
plt.yscale("log")
plt.ylim(10**-15, 10**0)
plt.xlim(1, 150)
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)
plt.legend(fontsize = legend_size)
plt.title(r'$\alpha$QPE; $\alpha = 1$', fontsize = 20)
plt.savefig("alpha-VQE/plots/aqpe_pt8.pdf", bbox_inches = 'tight')
