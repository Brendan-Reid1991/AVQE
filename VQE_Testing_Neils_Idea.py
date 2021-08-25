import numpy as np
import random

true_exp = random.uniform(-1, 1)
true_p0 = (1 + true_exp)/2

tolerance = 10**-4

accuracy = 100

est_exp = random.uniform(-1, 1)
est_p0 = 0
count = 0
while accuracy > tolerance:
    count += 1
    if random.uniform(0, 1) < true_p0:
        zero_out = 1
    else:
        zero_out = 0

    est_p0 += zero_out

    updated_est = -1 + 2*est_p0/count

    accuracy = abs(est_exp - updated_est)
    est_exp = updated_est

print(true_exp, updated_est)
