"""
Test the behavior of V1 as a function of the number of channels.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from most_queue.theory.fifo.mmnr import MMnrCalc


def linear_appoximation(channels, v1_array):
    """
    Find linear approximation, where xs are number of channels and ys are values of V1.
    """
    xs = np.array(channels)
    ys = np.array(v1_array)

    # Fit a linear model to the data
    slope, intercept = np.polyfit(xs, ys, 1)

    return slope, intercept


def test_v1_on_channels():
    """
    Test the behavior of V1 as a function of the number of channels.
    """
    # Define parameters

    utilizations = [0.5, 0.75, 0.9]  # Utilization levels to test
    channels = [x for x in range(1, 6)]  # Number of channels to test
    arrival_rate = 1

    _fig, ax = plt.subplots()

    for u in utilizations:
        v1_array = []
        for c in channels:
            # Calculate arrival rate based on utilization
            mu = arrival_rate / (u * c)
            mmnr_calc = MMnrCalc(n=c, r=100)
            mmnr_calc.set_sources(l=arrival_rate)
            mmnr_calc.set_servers(mu=mu)
            v1_array.append(mmnr_calc.get_v()[0])

        ax.plot(channels, v1_array, label=r"$\rho$" + f" ={u}")

        slope, intercept = linear_appoximation(channels, v1_array)
        ax.plot(
            channels,
            [slope * x + intercept for x in channels],
            label="approx " + r"$\rho$" + f" ={u}",
        )

    ax.set_xlabel("Channels")
    # set int ticks
    ax.set_xticks(np.arange(min(channels), max(channels) + 1, 1.0))
    ax.set_ylabel("v1")
    plt.legend()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # Replace with your desired path
    save_path = os.path.join(cur_dir, "v1_on_channels.png")
    plt.savefig(save_path)

    plt.close(_fig)


if __name__ == "__main__":
    test_v1_on_channels()
