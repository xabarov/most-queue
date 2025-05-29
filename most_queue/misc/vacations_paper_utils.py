"""
Utility functions for processing data from vacation-type queueing systems.
"""
import json
import math

import matplotlib.pyplot as plt
from most_queue.general.tables import probs_print, times_print


def print_table(experiments_stat):
    """
    Prints a table comparing the results of the Takahashi-Takami 
    method and simulation modeling for different queueing systems.
    :param experiments_stat: A list of dictionaries containing the results of the experiments.
    :type experiments_stat: list[dict]
    :return: None
    """
    for stat in experiments_stat:
        # Print header information
        print(
            f"\nComparison of M/H2/{stat['n']} with H2-warm-up, H2-cooling and H2-cooling delay")
        print(f"Utilization factor: {stat['ro']:0.2f}")
        # Service time coefficients of variation
        coeff_variants = [
            ("Service", stat["coev_service"]),
            ("Warm-up", stat["coev_warm"]),
            ("Cooling", stat["coev_cold"]),
            ("Cooling delay", stat["coev_cold_delay"])
        ]

        for label, value in coeff_variants:
            print(f"{label} time coefficient of variation {value:0.3f}")

        # Print probabilities
        states = ["warm", "cold", "cold_delay"]
        for state in states:
            tt_val = stat[f'tt_{state}_prob']
            sim_val = stat[f'sim_{state}_prob']
            print(f"\nProbability of being in a {state} state")
            print(f"\tSim: {sim_val:0.3f}\n\tCalc: {tt_val:0.3f}")

        # Print run times
        print("\nAlgorithm run times:")
        print(
            f"Running time of the Takahashi-Takami algorithm: {stat['tt_time']:0.3f} c")
        print(f"Simulation time: {stat['sim_time']:0.3f} c")

        probs_print(p_sim=stat['p_sim'], p_num=stat['p_tt'], size=10)
        times_print(sim_moments=stat["w_sim"], calc_moments=stat["w_tt"])


def dump_stat(experiments_stat, save_name='run_stat.json'):
    """
    Dump the results of experiments to a JSON file.
    :param experiments_stat: A list of dictionaries containing the results of the experiments.
    :param save_name: The name of the file to save the results to. Defaults to 'run_stat.json'.
    """
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(experiments_stat, f)


def load_stat(stat_name):
    """
    Load the results of experiments from a JSON file.
    :param stat_name: The name of the file to load the results from.
    :return: A list of dictionaries containing the results of the experiments.
    """
    with open(stat_name, 'r', encoding='utf-8') as f:
        return json.load(f)


def calc_moments_by_mean_and_coev(mean, coev):
    """
    Calculate the initial three moments (mean, variance, and skewness) 
    based on the mean and coefficient of variation.
    :param mean: The mean value of the distribution.
    :param coev: The coefficient of variation (standard deviation divided by the mean).
    :return: A list containing the calculated moments [mean, variance, skewness].
    """
    b = [0.0] * 3
    alpha = 1 / (coev ** 2)
    b[0] = mean
    b[1] = math.pow(b[0], 2) * (math.pow(coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)
    return b


def make_plot(experiments_stat, w_moments_num=0,
              param_name="ro", mode='error', save_path=None):
    """
    Build plot for wait times initial moments
    :param experiments_stat: list of experiment statistics
    :param w_moments_num: number of moment to plot (0 - mean, 1 - variance, 2 - skewness)
    :param param_name: name of parameter to use for x-axis
    :param mode: 'error' or 'mass' - whether to plot errors or mass values
    :return: figure and axis objects
    """
    _fig, ax = plt.subplots()
    w_sim_mass = []
    w_tt_mass = []
    xs = []
    if mode == 'error':
        errors = []

    for stat in experiments_stat:
        if mode == 'error':
            w_sim = stat["w_sim"][w_moments_num]
            w_tt = stat["w_tt"][w_moments_num]
            errors.append(100 * (w_sim - w_tt) / w_tt)
        else:
            w_sim_mass.append(stat["w_sim"][w_moments_num])
            w_tt_mass.append(stat["w_tt"][w_moments_num])

        if param_name != 'delay_mean':
            xs.append(stat[param_name])
        else:
            xs.append(stat["b_d"][0])

    if mode == 'error':
        ax.plot(xs, errors)
        ax.set_ylabel(r"$\varepsilon$, %")
    else:
        ax.plot(xs, w_sim_mass, label="Sim")
        ax.plot(xs, w_tt_mass, label="Calc")
        ax.set_ylabel(r"$\omega_{1}$")
        plt.legend()

    if param_name == 'ro':
        ax.set_xlabel(r"$\rho$")
    elif param_name == 'coev':
        ax.set_xlabel(r"$\nu$")
    elif param_name == 'n':
        ax.set_xlabel("n")
    elif param_name == "delay_mean":
        ax.set_xlabel("t, с")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
