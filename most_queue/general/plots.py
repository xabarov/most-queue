"""
Plotting utilities for simulation results.
"""
from enum import Enum

import matplotlib.pyplot as plt


class DependsType(Enum):
    """
    Enum to specify the type of dependency for plotting.
    """
    NONE = 0
    UTILIZATION_FACTOR = 1
    CHANNELS_NUMBER = 2
    COEFFICIENT_OF_VARIATION = 3


def plot_sim_vs_calc_moments(xs: list[float], sim_results: list[float],
                             calc_results: list[float],
                             depends_on: DependsType = DependsType.CHANNELS_NUMBER,
                             is_errors=False, is_waiting_time=True, save_path=None):
    """
    Plots the simulation and calculated moments for a given list of x values.
    :param xs: A list of x values (e.g., utilization factors or number of channels).
    :type xs: list[float]
    :param sim_results: A list of simulated results.
    :type sim_results: list[float]
    :param calc_results: A list of calculated results using some analytical or numerical method.
    :param depends_on: The type of dependency for plotting (utilization factor, number of channels, etc.).
    :type depends_on: DependsType
    :param is_errors: If True, plots the percentage error between simulation and calculation results.
    :type is_errors: bool
    :param is_waiting_time: If True, labels the y-axis as waiting time; otherwise, as sojourn time.
    :type is_waiting_time: bool
    :param save_path: The path where the plot should be saved. If None, the plot will not be saved.
    :type save_path: str or NoneType
    :return: None
    """
    _fig, ax = plt.subplots()
    if is_errors:
        errors = [100 * (w_sim - w_tt) / w_tt for w_sim,
                  w_tt in zip(sim_results, calc_results)]

        ax.plot(xs, errors, color='black')
        ax.set_ylabel(r"$\varepsilon$, %")
    else:
        ax.plot(xs, sim_results, label="Sim", color='black', linestyle='--')
        ax.plot(xs, calc_results, label="Calc", color='black')
        if is_waiting_time:
            ax.set_ylabel(r"$\omega_{1}$")
        else:
            ax.set_ylabel(r"$\upsilon_{1}$")
        plt.legend()

    if depends_on == DependsType.UTILIZATION_FACTOR:
        ax.set_xlabel(r"$\rho$")
    elif depends_on == DependsType.COEFFICIENT_OF_VARIATION:
        ax.set_xlabel(r"$\nu$")
    else:
        ax.set_xlabel("n")
        plt.xticks(range(1, len(xs) + 1))

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close(_fig)
