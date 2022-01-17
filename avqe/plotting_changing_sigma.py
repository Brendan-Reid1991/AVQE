from matplotlib import lines
import matplotlib.pyplot as plt
import pickle
import matplotlib
from avqe import AVQE
import numpy as np
from numpy import pi

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


sigma_values = [pi/16, pi/8, pi/4, pi/2]
alpha_values = np.linspace(0, 1, 11)
label = ["pi-16", "pi-8", "pi-4", "pi-2"]
plt_label = ["16", "8", "4", "2"]

errors = []
failures = []
for ele in label:
    f = open("avqe/data/avqe_sup_test_sigma_%s"%ele, "rb")
    data = pickle.load(f)
    errors.append(transpose(data)[1])
    failures.append(transpose(data)[3])

for idx, ele in enumerate(errors):
    plt.plot(alpha_values, ele, label = r'$\sigma=\pi/$'+'%s'%plt_label[idx])

plt.ylabel('Median error')
plt.xlabel(r'$\alpha$')
plt.legend(title=r'$\sigma$', title_fontsize = legend_title_size)
plt.savefig('sigma_errors.pdf', bbox_inches = 'tight')
plt.grid(True)
plt.clf()

for idx, ele in enumerate(failures):
    plt.plot(alpha_values, ele, label = r'$\sigma=\pi/$%s'%plt_label[idx])

plt.ylabel('Failure rate')
plt.xlabel(r'$\alpha$')
plt.legend(title=r'$\sigma$', title_fontsize = legend_title_size)
plt.grid(True)
plt.savefig('sigma_fails.pdf', bbox_inches = 'tight')
plt.clf()