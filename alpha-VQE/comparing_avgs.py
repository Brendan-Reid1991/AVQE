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


aqpe_data = []
rfpe_data= [] 
aqpe_exact = []
rfpe_exact = []
for _ in range(1000):
    phi = np.random.uniform(-pi,pi)

    rfpe_init = rfpe(phi=phi, sigma=pi/2, nSamples=400, alpha=1, update=1)
    rfpe_get_res = rfpe_init.run_experiment(exp_num)
    rfpe_data.append(rfpe_get_res)

    aqpe_init = aqpe(phi=phi, sigma=pi/2, nSamples=400, alpha=1, update=1)
    aqpe_get_res = aqpe_init.run_experiment(exp_num)
    aqpe_data.append(aqpe_get_res)

    exact_aqpe = aqpe(phi=phi, sigma=pi/2, nSamples=1, alpha = 1, update = 2)
    exact_res = exact_aqpe.run_experiment(exp_num) 
    aqpe_exact.append(exact_res)

    exact_rfpe = rfpe(phi=phi, sigma=pi/2, nSamples=1, alpha = 1, update = 2)
    exact_res = exact_rfpe.run_experiment(exp_num) 
    rfpe_exact.append(exact_res)

aqpe_avgd = [np.mean(x) for x in transpose(aqpe_data)]
rfpe_avgd = [np.mean(x) for x in transpose(rfpe_data)]

aqpe_exact_avgd = [np.mean(x) for x in transpose(aqpe_exact)]
rfpe_exact_avgd = [np.mean(x) for x in transpose(rfpe_exact)]

plt.plot(range(exp_num), rfpe_avgd, label = "k=400")
plt.plot(range(exp_num), rfpe_exact_avgd, '--', color = 'grey', alpha = 0.5, label = 'Exact' )

plt.xlabel("Experiment",fontsize = label_size)
plt.ylabel("Mean Error",fontsize = label_size)
plt.grid(True)
plt.yscale("log")
plt.ylim(10**-15, 10**0)
plt.xlim(1, 150)
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)
plt.legend(fontsize = legend_size)
plt.title('RFPE', fontsize = 20)
plt.savefig("alpha-VQE/plots/rfpe_mean.pdf", bbox_inches = 'tight')
plt.clf()



plt.plot(range(exp_num), aqpe_avgd, label = "k=400")
plt.plot(range(exp_num), aqpe_exact_avgd, '--', color = 'grey', alpha = 0.5, label = 'Exact' )

plt.xlabel("Experiment",fontsize = label_size)
plt.ylabel("Mean Error",fontsize = label_size)
plt.grid(True)
plt.yscale("log")
plt.ylim(10**-15, 10**0)
plt.xlim(1, 150)
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)
plt.legend(fontsize = legend_size)
plt.title(r'$\alpha$QPE; $\alpha = 1$', fontsize = 20)
plt.savefig("alpha-VQE/plots/aqpe_mean.pdf", bbox_inches = 'tight')



