import matplotlib.pyplot as plt
import pickle
import matplotlib


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

bare = open("data_test_avqe", "rb")
data = pickle.load(bare)


max_depths, errs, runs = transpose(data)

plt.plot(max_depths, errs, linewidth = 2, color = colours[0])
plt.plot(max_depths, [0.005]*len(errs), '--', linewidth = 1.5, color = 'gray')
plt.xticks(size = tick_size)
plt.yticks(size = tick_size)

plt.ylabel(r'$|\phi - \mu|^2$', fontsize = label_size)
plt.xlabel(r'$D$', fontsize = label_size)
# plt.legend(fontsize = legend_size)
# plt.xlim(0, 1)

plt.grid(True)
plt.savefig('max_depth_v_err.png', bbox_inches='tight')
plt.clf()

