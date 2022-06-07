import numpy as np
import math

coevs = np.linspace(0.7, 5, 20)

for coev in coevs:
    a = [1, 0, 0]
    alpha = 1 / (coev ** 2)
    a[1] = math.pow(a[0], 2) * (math.pow(coev, 2) + 1)
    a[2] = a[1] * a[0] * (1.0 + 2 / alpha)
    print("{0:.2f}, {1:.2f}, {2:.2f}".format(*a))
