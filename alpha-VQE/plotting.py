from matplotlib import lines
import matplotlib.pyplot as plt
import pickle
import matplotlib
from alpha_vqe import Alpha_VQE

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

bare = open("data_test", "rb")
data = pickle.load(bare)

alphas, errs, runs = transpose(data)

max_shots = []
for al in alphas:
    ms = Alpha_VQE(phi = 0, alpha = al, nSamples=100).get_max_shots()
    max_shots.append(ms)

plt.plot(alphas, errs, linewidth = 2, color = colours[0])
plt.plot(alphas, [0.005]*len(errs), '--', linewidth = 1.5, color = 'gray')
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)



plt.ylabel(r'$|\phi - \mu|^2$', fontsize = label_size)
plt.xlabel(r'$\alpha$', fontsize = label_size)
plt.legend(fontsize = legend_size)
plt.xlim(0, 1)

plt.grid(True)
plt.savefig('alpha_vqe_vs_error.png', bbox_inches='tight')
plt.clf()

plt.plot(alphas, runs, linewidth = 2, color = colours[0], label = "Numerics")
plt.plot(alphas, max_shots, linewidth = 2, color = colours[1], label = "Theory")
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

plt.yscale("log")

plt.ylabel(r'$N_{runs}$', fontsize = label_size)
plt.xlabel(r'$\alpha$', fontsize = label_size)
plt.legend(fontsize = legend_size)
plt.xlim(0, 1)

plt.grid(True)
plt.savefig('alpha_vqe_vs_runs.png', bbox_inches='tight')
plt.clf()