"""
Test Weibull distribution
"""
import matplotlib.pyplot as plt
import numpy as np

from most_queue.rand_distribution import Weibull

if __name__ == "__main__":

    MEAN = 1.0
    coevs = [1.0, 2.1, 3.2]

    fig, ax = plt.subplots()

    for coev in coevs:
        params = Weibull.get_params_by_mean_and_coev(MEAN, coev)

        print(
            f"Weibull params at CV= {coev:1.3f}: k = {params.k:1.3f} W = {params.W:1.3f}")
        ts = np.linspace(0, 3 * coevs[len(coevs)-1], 100)
        dfr = [Weibull.get_tail(params, t) for t in ts]

        ax.plot(ts, dfr, label=f"$\\nu$  = {coev:1.1f}")

    ax.set_xlabel('t')
    ax.set_ylabel('Tail')
    plt.legend()
    plt.show()
