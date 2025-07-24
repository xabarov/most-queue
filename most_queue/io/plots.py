"""
Plotting utilities for simulation results.
"""

from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class DependsType(Enum):
    """
    Enum to specify the type of dependency for plotting.
    """

    NONE = 0
    UTILIZATION_FACTOR = 1
    CHANNELS_NUMBER = 2
    COEFFICIENT_OF_VARIATION = 3


class Plotter:
    """
    A class for plotting simulation results.
    """

    def __init__(
        self,
        xs: list[float],
        sim_results: list[float],
        calc_results: list[float],
        depends_on: DependsType = DependsType.CHANNELS_NUMBER,
    ):
        self.xs = xs
        self.sim_results = sim_results
        self.calc_results = calc_results
        self.depends_on = depends_on

    def _configure_plot(self, ax: Axes, ylabel: str):
        """
        Helper method to configure common plot elements.
        """
        ax.set_ylabel(ylabel)
        if self.depends_on == DependsType.UTILIZATION_FACTOR:
            ax.set_xlabel(r"$\rho$")
        elif self.depends_on == DependsType.COEFFICIENT_OF_VARIATION:
            ax.set_xlabel(r"$\nu$")
        else:
            ax.set_xlabel("n")
            plt.xticks(range(1, len(self.xs) + 1))

    def plot_errors(self, save_path: str = None):
        """
        Plots the simulation and calculated moments for a given list of x values.
        """
        _fig, ax = plt.subplots()
        errors = [100 * (w_sim - w_tt) / w_tt for w_sim, w_tt in zip(self.sim_results, self.calc_results)]

        ax.plot(self.xs, errors, color="black")
        self._configure_plot(ax, r"$\varepsilon$, %")

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

        plt.close(_fig)

    def plot_waiting(self, save_path: str = None):
        """
        Plots the simulation and calculated queue waiting times.
        """
        _fig, ax = plt.subplots()
        ax.plot(self.xs, self.sim_results, label="Sim", color="black", linestyle="--")
        ax.plot(self.xs, self.calc_results, label="Calc", color="black")
        ax.legend()
        self._configure_plot(ax, r"$\omega_{1}$")

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

        plt.close(_fig)

    def plot_sojourn(self, save_path: str = None):
        """
        Plots the simulation and calculated sojourn times.
        """
        _fig, ax = plt.subplots()
        ax.plot(self.xs, self.sim_results, label="Sim", color="black", linestyle="--")
        ax.plot(self.xs, self.calc_results, label="Calc", color="black")
        ax.legend()
        self._configure_plot(ax, r"$\upsilon_{1}$")

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

        plt.close(_fig)
