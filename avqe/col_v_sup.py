from matplotlib import lines
import matplotlib.pyplot as plt
import pickle
import matplotlib
from avqe import AVQE
import numpy as np

# Font choices to match with LaTeX as close as possible
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'Franklin Gothic Book Regular'
matplotlib.rcParams['axes.linewidth'] = 1.2
colours=['#1d3c34','#00968f','#3ccbda', '#ff7500','#dc4405']

tick_size = 15
label_size = 20
legend_title_size = 15
legend_size = 15

def transpose(L):
    return(list(map(list,zip(*L))))


bare = open("avqe/data/avqe_sup_test", "rb")
sup = pickle.load(bare)

bare = open("avqe/data/avqe_col_test", "rb")
col = pickle.load(bare)

depths, s_errs, s_runs, s_failures = transpose(sup)
depths1, c_errs, c_runs, c_failures = transpose(col)

# failure_pct = []
# for val in failures:
#     f_pct = 100*val / (val + 1000) 
#     failure_pct.append(f_pct)

alpha_values = np.linspace(0, 1, 11)
max_shots = [] 
for al in alpha_values:
    md = np.ceil(1/0.005**al)
    ms = AVQE(phi = 0, max_unitaries = md).get_max_shots()
    max_shots.append(ms)

plt.plot(depths, s_errs, linewidth = 2, color = colours[0], label = "Superposition")
plt.plot(depths, c_errs, linewidth = 2, color = colours[1], label = "Collapsed")

plt.yticks(size = tick_size)



plt.ylabel(r'Median Error', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
plt.xscale('log')
plt.xticks(size = tick_size)

plt.ylim(0, 0.0015)
plt.legend(fontsize = legend_size)


plt.grid(True)
plt.savefig('avqe/plots/avqe_vs_error_compare.pdf', bbox_inches='tight')
plt.clf()

plt.plot(depths, s_runs, linewidth = 2, color = colours[0], label = "Superposition")
plt.plot(depths, c_runs, linewidth = 2, color = colours[1], label = "Collapsed")
plt.xticks(depths, size = tick_size)
plt.yticks(size = tick_size)



plt.ylabel(r'$N_\mathrm{runs}$', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
plt.legend(fontsize = legend_size)

plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.savefig('avqe/plots/avqe_vs_runs_compare.pdf', bbox_inches='tight')
plt.clf()


plt.plot(depths, np.asarray(s_runs) / np.asarray(c_runs), linewidth = 2, color = colours[0])
plt.xticks(depths, size = tick_size)
plt.yticks(size = tick_size)



plt.ylabel(r'Sup / Col ratio of measurements', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)


# plt.xscale('log')
plt.grid(True)
plt.savefig('avqe/plots/avqe_vs_runs_ratios.pdf', bbox_inches='tight')
plt.clf()