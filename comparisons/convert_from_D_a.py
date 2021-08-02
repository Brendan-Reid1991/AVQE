import numpy as np

alpha_values = np.linspace(0, 1, 21)
d_values = [np.ceil(1/0.005**a) for a in alpha_values]
p = 0.005

print(np.ceil(1.25/p**0))
print(alpha_values[::2])
print(d_values[::2])

