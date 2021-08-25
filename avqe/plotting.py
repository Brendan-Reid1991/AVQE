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


bare = open("avqe/data/avqe_col_err_run_v_alpha", "rb")
data = pickle.load(bare)

depths, errs, runs, failures = transpose(data)

failure_pct = []
for val in failures:
    f_pct = 100*val / (val + 1000) 
    failure_pct.append(f_pct)

alpha_values = np.linspace(0, 1, 21)
max_shots = []
for al in alpha_values:
    md = np.ceil(1/0.005**al)
    ms = AVQE(phi = 0, max_unitaries = md).get_max_shots()
    max_shots.append(ms)

plt.plot(depths, errs, linewidth = 2, color = colours[0])
plt.plot(depths, [0.005]*len(errs), '--', linewidth = 1.5, color = 'gray')
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)



plt.ylabel(r'$|\phi - \mu|^2$', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
# plt.legend(fontsize = legend_size)


plt.grid(True)
plt.savefig('avqe/plots/avqe_vs_error.png', bbox_inches='tight')
plt.clf()

plt.plot(depths, runs, linewidth = 2, color = colours[0], label = "Numerics")
plt.plot(depths, max_shots, linewidth = 2, color = colours[1], label = "Theory")
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

plt.yscale("log")

plt.ylabel(r'$N_{runs}$', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
plt.legend(fontsize = legend_size)

plt.grid(True)
plt.savefig('avqe/plots/avqe_vs_runs.png', bbox_inches='tight')
plt.clf()



plt.plot(depths, failure_pct, linewidth = 2, color = colours[0])
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

# plt.yscale("log")

plt.ylabel(r'Failure Rate %', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
# plt.legend(fontsize = legend_size)

plt.grid(True)
plt.savefig('avqe/plots/avqe_failures.png', bbox_inches='tight')
plt.clf()

sigma_errs = []
sigma_runs = []
sigma_failures = []

fracs = [2,4,8,16]

for x in fracs:
    bare = open("avqe/data/avqe_failure_sig_pi-%s"%x, "rb")

    data = pickle.load(bare)
    _void, errs, runs, fails = transpose(data)

    sigma_errs.append(errs)
    sigma_runs.append(runs)
    sigma_failures.append(fails)

for idx, ele in enumerate(sigma_errs):
    plt.plot(depths, ele, linewidth = 2, color = colours[idx], label = r"$\pi$"+"/%s"%fracs[idx])
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

plt.ylabel(r'$|\phi - \mu|$', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
plt.legend(fontsize = legend_size)

plt.ylim(5*10**-4, 1.5*10**-3)

plt.grid(True)
plt.savefig('avqe/plots/avqe_changing_sigma_errs.png', bbox_inches='tight')
plt.clf()

for idx, ele in enumerate(sigma_runs):

    plt.plot(depths, ele, linewidth = 2, color = colours[idx], label = r"$\pi$"+"/%s"%fracs[idx])
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

plt.ylabel(r'$N_{runs}$', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
plt.legend(fontsize = legend_size)

# plt.ylim(5*10**-4, 1.5*10**-3)

plt.grid(True)
plt.savefig('avqe/plots/avqe_changing_sigma_runs.png', bbox_inches='tight')
plt.clf()

for idx, ele in enumerate(sigma_failures):
    
    plt.plot(depths, ele, linewidth = 2, color = colours[idx], label = r"$\pi$"+"/%s"%fracs[idx])
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

plt.ylabel(r'$N_{runs}$', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
plt.legend(fontsize = legend_size)

# plt.ylim(5*10**-4, 1.5*10**-3)

plt.grid(True)
plt.savefig('avqe/plots/avqe_changing_sigma_fails.png', bbox_inches='tight')
plt.clf()