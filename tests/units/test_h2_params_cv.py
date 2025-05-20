"""
Plot the parameters of the H2-distribution for a given set of initial moments.
"""

import matplotlib.pyplot as plt
import numpy as np

from most_queue.rand_distribution import H2Distribution


def plot_h2_params_less_than_exp(start_cv=0.5, end_cv=0.95, num_points=100, y_ax_threshold=3):
    """
    Plot the parameters of the H2-distribution for a given set of initial moments,
      that are less than the exponential distribution.
    :param start_cv: The starting coefficient of variation for the initial moments.
    :param end_cv: The ending coefficient of variation for the initial moments.
    :param num_points: The number of points to plot.
    :param y_ax_threshold: The threshold for the y-axis. 
    If the value is greater than this, it will not be plotted.
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    cv_values = np.linspace(start_cv, end_cv, num_points)
    mu1_values_real = []
    mu2_values_real = []
    p1_values_real = []
    mu1_values_imag = []
    mu2_values_imag = []
    p1_values_imag = []
    for cv in cv_values:
        h2_params = H2Distribution.get_params_by_mean_and_coev(
            1, cv, is_clx=True)
        mu1_values_real.append(h2_params.mu1.real)
        mu1_values_imag.append(h2_params.mu1.imag)
        mu2_values_real.append(h2_params.mu2.real)
        mu2_values_imag.append(h2_params.mu2.imag)
        p1_values_real.append(h2_params.p1.real)
        p1_values_imag.append(h2_params.p1.imag)

    # plot mu1
    ax.plot(cv_values, mu1_values_real, label='mu1_real',
            color='orange', linestyle='--')
    ax.plot(cv_values, mu1_values_imag, label='mu1_imag',
            linestyle='-.', color='red')

    # plot mu2
    ax.plot(cv_values, mu2_values_real, label='mu2_real',
            color='blue', linestyle='dotted')
    ax.plot(cv_values, mu2_values_imag, label='mu2_imag',
            linestyle='dashed', color='blue')
    # plot p1
    ax.plot(cv_values, p1_values_real, label='p1_real', color='green')
    ax.plot(cv_values, p1_values_imag, label='p1_imag',
            linestyle='-.', color='green')

    ax.set_xlabel('Coefficient of Variation (CV)')
    ax.set_ylabel('Parameter Value')

    # threshold y axis
    ax.set_ylim(-y_ax_threshold, y_ax_threshold)
    ax.set_title('H2-distribution Parameters vs. CV')
    plt.legend()
    plt.savefig('tests/units/h2_params_cv_less_than_exp.png')
    plt.close(fig)


def plot_h2_params_greater_than_exp(start_cv=1.01, end_cv=5.0, num_points=100):
    """
    Plot the parameters of the H2-distribution for a given set of initial moments, 
    that are greater than exponential.
    """
    cv_values = np.linspace(start_cv, end_cv, num_points)
    mu1_values = []
    mu2_values = []
    p1_values = []
    for cv in cv_values:
        h2_params = H2Distribution.get_params_by_mean_and_coev(
            1, cv, is_clx=False)
        mu1_values.append(h2_params.mu1)
        mu2_values.append(h2_params.mu2)
        p1_values.append(h2_params.p1)

    # plot mu1
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cv_values, mu1_values, label='mu1', color='red')

    # plot mu2
    ax.plot(cv_values, mu2_values, label='mu2', color='blue')
    # plot p1
    ax.plot(cv_values, p1_values, label='p1', color='green')

    ax.set_xlabel('Coefficient of Variation (CV)')
    ax.set_ylabel('Parameter Value')
    ax.set_title('H2-distribution Parameters vs. CV')
    plt.legend()
    plt.savefig('tests/units/h2_params_cv_greater_than_exp.png')
    plt.close(fig)


if __name__ == "__main__":
    plot_h2_params_less_than_exp()
    plot_h2_params_greater_than_exp()
