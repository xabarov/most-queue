"""
Class for calculating M/G/1 queue with disasters.
Use results from the paper:
    Jain, Gautam, and Karl Sigman. "A Pollaczek–Khintchine formula for M/G/1 queues with disasters."
    Journal of Applied Probability 33.4 (1996): 1191-1200.
"""

import numpy as np
from scipy.misc import derivative

from most_queue.rand_distribution import GammaDistribution, H2Distribution
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.busy_periods import calc_busy_pls
from most_queue.theory.utils.transforms import lst_gamma, lst_h2


class MG1Disasters(BaseQueue):
    """
    Class for calculating M/G/1 queue with disasters.
    """

    def __init__(
        self,
        calc_params: CalcParams | None = None,
    ):
        """
        Initialize the MG1Disasters class.
        :param calc_params: Calculation parameters. If None, default parameters are used.
        """

        super().__init__(n=1, calc_params=calc_params)
        self.l_pos = None
        self.l_neg = None
        self.b = None
        self.approximation = self.calc_params.approx_distr

        self.lst_function = None
        self.params = None

        self.nu = None

    def set_sources(self, l_pos: float, l_neg: float):  # pylint: disable=arguments-differ
        """
        Set the arrival rates of positive and negative jobs
        :param l_pos: arrival rate of positive jobs
        :param l_neg: arrival rate of negative jobs
        """
        self.l_pos = l_pos
        self.l_neg = l_neg
        self.is_sources_set = True

    def set_servers(self, b: list[float]):  # pylint: disable=arguments-differ
        """
        Set the initial moments of service time distribution
        :param b: initial moments of service time distribution
        """
        self.b = b

        if self.approximation == "h2":
            self.lst_function = lst_h2
            self.params = H2Distribution.get_params(b)
        elif self.approximation == "gamma":
            self.lst_function = lst_gamma
            self.params = GammaDistribution.get_params(b)
        else:
            raise ValueError("Approximation must be 'h2' or 'gamma'.")
        self.is_servers_set = True

    def _calc_nu(self):
        """
        Calculate the nu parameter.
        Notice: l_neg*E[min{B, Y}] = 1-b_lst(l_neg)),
        so nu = E[min{B, Y}]/(1/l_neg + E[min{B, Y}]) =
        (1-b_lst(l_neg))/(self.l_neg/self.l_pos + 1 - b_lst(l_neg))
        """

        busy_s = calc_busy_pls(self.lst_function, self.params, self.l_pos, self.l_neg)
        x = self.l_pos * (1 - busy_s)
        nu = x / (self.l_neg + x)
        return nu

    def _v_lst(self, s):
        """
        Calculate  Laplace-Stieljets transform for waiting time in the system.
        """

        self.nu = self.nu or self._calc_nu()
        numerator = s * (1 - self.nu) - self.l_neg
        big_g = self.lst_function(self.params, s)
        denominator = s - self.l_pos * (1.0 - big_g) - self.l_neg
        return numerator / denominator

    def get_v(self) -> list[float]:
        """
        Calculate first three moments of sojourn time in the system.
        """
        self._check_if_servers_and_sources_set()
        v = [0, 0, 0]
        for i in range(3):
            v[i] = derivative(self._v_lst, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
        v = np.array([-v[0], v[1].real, -v[2]])

        return v
