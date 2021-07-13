from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib
from alpha_vqe import Alpha_VQE

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

alpha_values = np.linspace(0, 1, 21)
sample_sizes = [100, 250, 500, 750, 1000]

sample_size_errs = []
sample_size_runs = []
for ss in sample_sizes:
    in1 = open("alpha-VQE/data/alpha_vqe_ss%s_alpha"%ss, "rb")
    in2 = pickle.load(in1)
    errs, runs = transpose(in2)[1], transpose(in2)[2]

    sample_size_errs.append(errs)
    sample_size_runs.append(runs)

in_exact = open("alpha-VQE/data/alpha_exact_alpha", "rb")
in_exact_loaded = pickle.load(in_exact)

errs_exact, runs_exact = transpose(in_exact_loaded)[0], transpose(in_exact_loaded)[1]

plt.plot(alpha_values, errs_exact, linewidth = 2, color = "gray", label = "Exact")
for idx, ss in enumerate(sample_sizes):
    plt.plot(alpha_values, sample_size_errs[idx], linewidth = 2, color = colours[idx], label = "%s"%ss)



plt.grid(True)
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)
plt.ylabel(r'$|\phi - \mu|$', fontsize = label_size)
plt.xlabel(r'$\alpha$', fontsize = label_size)
plt.legend(fontsize = legend_size, title = "Sample Sizes")
plt.savefig("alpha-VQE/plots/Increasing_Sample_Size_Errors.pdf", bbox_inches = 'tight')
plt.clf()


def eq1(pre, alpha):
    if alpha < 1:
        xa = (2/(1-alpha))*(1/(pre**(2-2*alpha)) - 1)
    else:
        xa = 4*np.log(1/pre)
    return(xa)

formula = []
for al in alpha_values:
    formula.append(
        eq1(0.005, al)
    )

fig, ax = plt.subplots(figsize=[5, 4])
ax.plot(alpha_values, runs_exact, linewidth = 2, color = "gray", label = "Exact")
ax.plot(alpha_values, formula, '--', linewidth = 1.5, color = "gray", label = "EQ1", alpha = 0.75)
ax.grid(True)
for idx, ss in enumerate(sample_sizes):
    ax.plot(alpha_values, sample_size_runs[idx], linewidth = 2, color = colours[idx], label = "%s"%ss)

axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])

axins.plot(alpha_values, runs_exact, linewidth = 2, color = "gray", label = "Exact")


for idx, ss in enumerate(sample_sizes):
    axins.plot(alpha_values, sample_size_runs[idx], linewidth = 2, color = colours[idx], label = "%s"%ss)


x_ticks = (0, 0.25, 0.5, 0.75, 1)
y_ticks = (0, 1000,2000,3000,4000,5000)
# exit()
axins.grid(True)
axins.set_xlim(0, 1)
axins.set_ylim(0, 5000)
# axins.set_xticks(x_ticks)
# axins.set_yticks(y_ticks)
# axins.set_xticklabels('')
# axins.set_yticklabels('')

plt.grid(True)
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)
plt.ylabel(r'No. of Updates', fontsize = label_size)
plt.xlabel(r'$\alpha$', fontsize = label_size)
plt.legend(fontsize = legend_size, title = "Sample Sizes", title_fontsize = legend_title_size, bbox_to_anchor=(1.05, 1), loc='upper left')

fig.savefig("alpha-VQE/plots/Increasing_Sample_Size_Runs.pdf", bbox_inches = 'tight')