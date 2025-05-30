
"""
Utils module for vacations paper
"""
import math

import matplotlib.pyplot as plt
import numpy as np


def calc_rel_error_percent(sim_value: float, calc_value: float) -> float:
    """Calculate the relative error percent between a simulation value and a calculated value."""
    return 100*(sim_value - calc_value) / sim_value if sim_value != 0 else np.inf


def calc_moments_by_mean_and_coev(mean, coev):
    """
    Calculate the E[X^k] for k=0,1,2
    for a distribution with given mean and coefficient of variation.
    :param mean: The mean value of the distribution.
    :param coev: The coefficient of variation (standard deviation divided by the mean).
    :return: A list containing the calculated moments
    """
    b = [0.0] * 3
    alpha = 1 / (coev ** 2)
    b[0] = mean
    b[1] = math.pow(b[0], 2) * (math.pow(coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)
    return b


def plot_w1(xs, w1_num, w1_sim, x_label: str, save_path=None, is_xs_int=False):
    """
    Plot the wait time averages for both calculation and simulation.
    :param xs: list of x-axis values.
    :param x_label: The label for the x-axis
    :param w1_num: The calculated wait time average.
    :param w1_sim: The simulated wait time average.
    :param save_path: The path to save the plot.
    :param is_xs_int: Whether the x-axis values are integers.
    """
    _fig, ax = plt.subplots()
    # plot first with dash -- line, black, second - with line, black
    ax.plot(xs, w1_num, label="num", color="black", linestyle="--")
    ax.plot(xs, w1_sim, label="sim", color="black")
    ax.legend()
    ax.set_xlabel(x_label)
    # set xticks to be integers
    if is_xs_int:
        ax.set_xticks(np.arange(min(xs), max(xs)+1, 1.0))
    ax.set_ylabel(r"$\omega_{1}$")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close(_fig)


def plot_w1_errors(xs, w1_rel_errors, x_label, save_path=None, is_xs_int=False):
    """
    Plot the wait time relative errors
    :param xs: The x-axis values.
    :param w1_rel_errors: The relative errors of omega_1.
    :param x_label: The label for the x-axis.
    :param save_path: The path to save the plot.
    :param is_xs_int: Whether the x-axis values are integers.
    """
    _fig, ax = plt.subplots()
    ax.plot(xs, w1_rel_errors, color="black")
    ax.set_xlabel(x_label)
    # set xticks to be integers
    if is_xs_int:
        ax.set_xticks(np.arange(min(xs), max(xs)+1, 1.0))
    ax.set_ylabel(r"$\varepsilon$, %")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close(_fig)


def plot_probs(xs, probs_num, probs_sim, x_label, save_path=None):
    """
    Plot the probabilities of different states
    :param xs: The x-axis values.
    :param probs_num: The probabilities from numerical calculation.
    :param probs_sim: The probabilities from simulation.
    :param x_label: The label for the x-axis.
    :param save_path: The path to save the plot.
    """
    _fig, ax = plt.subplots()
    ax.plot(xs, probs_num, label="num", color="black", linestyle="--")
    ax.plot(xs, probs_sim, label="sim", color="black")
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close(_fig)
