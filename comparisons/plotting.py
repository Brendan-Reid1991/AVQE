from matplotlib import lines
import matplotlib.pyplot as plt
import pickle
import matplotlib
from avqe import AVQE
import numpy as np

# Font choices to match with LaTeX as close as possible
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Franklin Gothic Book'
matplotlib.rcParams['axes.linewidth'] = 1.2
colours=['#1d3c34','#00968f','#3ccbda', '#ff7500','#dc4405']

tick_size = 15
label_size = 20
legend_title_size = 15
legend_size = 15

def transpose(L):
    return(list(map(list,zip(*L))))

avqe = open("avqe/data/avqe_test_acc_0.005", "rb")
avqe_data = pickle.load(avqe)
avqe_depths, avqe_errs, avqe_runs, avqe_failures = transpose(avqe)

alpha_vqe = open("alpha_vqe/data/alpha_vqe_ss1000_alpha", "rb")
alpha_vqe_data = pickle.load(alpha_vqe)
_void, alpha_errs, alpha_runs = transpose(alpha_vqe_data)




avqe_failure_pct = []
for val in avqe_failures:
    f_pct = 100*val / (val + 1000) 
    avqe_failure_pct.append(f_pct)

alpha_values = np.linspace(0, 1, 21)
max_shots = []
for al in alpha_values:
    md = np.ceil(1/0.005**al)
    ms = AVQE(phi = 0, max_unitaries = md).get_max_shots()
    max_shots.append(ms)

plt.plot(alpha_values, avqe_errs, linewidth = 2, color = colours[0])
plt.plot(alpha_values, avqe_errs, linewidth = 2, color = colours[0])
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)



plt.ylabel(r'$|\phi - \mu|^2$', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
# plt.legend(fontsize = legend_size)


plt.grid(True)
plt.savefig('avqe/plots/avqe_vs_error.png', bbox_inches='tight')
plt.clf()
