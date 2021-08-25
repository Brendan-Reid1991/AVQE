from matplotlib import lines
import matplotlib.pyplot as plt
import pickle
import matplotlib
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

avqe = open("avqe/data/avqe_col_err_run_v_alpha", "rb")
avqe_data = pickle.load(avqe)
avqe_depths, avqe_errs, avqe_runs, avqe_failures = transpose(avqe_data)

alpha_vqe = open("alpha-VQE/data/alpha_exact_alpha", "rb")
alpha_vqe_data = pickle.load(alpha_vqe)
alpha_errs, alpha_runs, alpha_failures = transpose(alpha_vqe_data)

alpha_values = np.linspace(0, 1, 21)

plt.plot(alpha_values, avqe_errs, linewidth = 2, color = colours[0], label = "AVQE")
plt.plot(alpha_values, alpha_errs, linewidth = 2, color = colours[1], label = r'$\alpha$VQE - Exact')
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

plt.ylim(0.0005, 0.0015)

plt.ylabel(r'$|\phi - \mu|^2$', fontsize = label_size)
plt.xlabel(r'$\alpha$', fontsize = label_size)
plt.legend(fontsize = legend_size)


plt.grid(True)
plt.savefig('comparisons/plots/avqe_vs_alpha_error.png', bbox_inches='tight')
plt.clf()

plt.plot(alpha_values[::2], avqe_runs[::2], linewidth = 2, color = colours[0], label = "AVQE")
plt.plot(alpha_values[::2], alpha_runs[::2], linewidth = 2, color = colours[1], label = r'$\alpha$VQE - Exact')
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

# plt.xlim(0.45, 0.55)
# plt.ylim(0, 2000)

plt.ylabel(r'$N_{runs}$', fontsize = label_size)
plt.xlabel(r'$\alpha$', fontsize = label_size)
plt.legend(fontsize = legend_size)
plt.yscale("log")

plt.grid(True)
plt.savefig('comparisons/plots/avqe_vs_alpha_runs.png', bbox_inches='tight')
plt.clf()

plt.plot(alpha_values[::2], np.asarray(avqe_runs[::2])/np.asarray(alpha_runs[::2]), linewidth = 2, color = colours[2])
# plt.plot(alpha_values[::2], , linewidth = 2, color = colours[1], label = r'$\alpha$VQE - Exact')
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

# plt.xlim(0.45, 0.55)
# plt.ylim(0, 2000)

plt.ylabel(r'$N_{runs}$', fontsize = label_size)
plt.xlabel(r'$\alpha$', fontsize = label_size)
# plt.legend(fontsize = legend_size)


plt.grid(True)
plt.savefig('comparisons/plots/ratio.png', bbox_inches='tight')
plt.clf()