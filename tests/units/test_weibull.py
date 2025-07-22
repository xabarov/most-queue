"""
Test Weibull distribution
"""

import matplotlib.pyplot as plt
import numpy as np

from most_queue.random.distributions import Weibull

if __name__ == "__main__":

    MEAN = 1.0
    cvs = [1.0, 2.1, 3.2]

    fig, ax = plt.subplots()

    for cv in cvs:
        params = Weibull.get_params_by_mean_and_cv(MEAN, cv)

        print(
            f"Weibull params at CV= {
                cv:1.3f}: k = {
                params.k:1.3f} W = {
                params.W:1.3f}"
        )
        ts = np.linspace(0, 3 * cvs[len(cvs) - 1], 100)
        dfr = [Weibull.get_tail(params, t) for t in ts]

        ax.plot(ts, dfr, label=f"$\\nu$  = {cv:1.1f}")

    ax.set_xlabel("t")
    ax.set_ylabel("Tail")
    plt.legend()
    plt.show()
