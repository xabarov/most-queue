"""
Class for calculating M/G/1 queue with warm-up.
"""

import time

import numpy as np
from scipy.misc import derivative

from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.transforms import lst_gamma, lst_h2


class MG1WarmCalc(BaseQueue):
    """
    Class for calculating M/G/1 queue with warm-up.
    """

    def __init__(self, calc_params: CalcParams | None = None):
        """
        Initialize the MG1WarmCalc class with arrival rate l,
        service time raw moments b, and warm-up service time moments b_warm.
        Parameters:
        l (float): Arrival rate.
        b (list[float]): raw moments of service time distribution.
        b_warm (list[float]): Warm-up moments of service time distribution.
        """

        super().__init__(n=1, calc_params=calc_params)

        self.l = None
        self.b = None
        self.b_warm = None
        self.b_param = None
        self.b_warm_param = None
        self.lst = None
        self.p0_star = None

        self.approximation = self.calc_params.approx_distr

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set the arrival rate
        """
        self.l = l
        self.is_sources_set = True

    def set_servers(self, b: list[float], b_warm: list[float]):  # pylint: disable=arguments-differ
        """
        Set the raw moments of service time distribution and warming time distribution

        :param b: raw moments of service time distribution
        :param b_warm: raw moments of warming time distribution
        """
        self.b = b
        self.b_warm = b_warm

        if self.approximation == "gamma":
            self.b_param = GammaDistribution.get_params(self.b)
            self.b_warm_param = GammaDistribution.get_params(self.b_warm)
            self.lst = lst_gamma
        elif self.approximation == "h2":
            self.b_param = H2Distribution.get_params(self.b)
            self.b_warm_param = H2Distribution.get_params(self.b_warm)
            self.lst = lst_h2
        else:
            raise ValueError("Invalid approximation method. Must be 'gamma' or 'h2'.")

        self.is_servers_set = True

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """
        Run calculations
        """

        start = time.process_time()

        v = self.get_v(num_of_moments)

        utilization = self.l * self.b[0]

        return QueueResults(v=v, utilization=utilization, duration=time.process_time() - start)

    def get_v(self, num_of_moments: int = 4) -> list[float]:
        """
        Calculate sourjourn moments for M/G/1 queue with warm-up.
        """

        if not self.v is None:
            return self.v

        tv = self.b_warm[0] / (1 - self.l * self.b[0])
        self.p0_star = 1 / (1 + self.l * tv)

        self._check_if_servers_and_sources_set()
        v = [0] * num_of_moments

        for i in range(num_of_moments):
            v[i] = derivative(self._calc_v_lst, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                v[i] = -v[i]
        self.v = np.array(v)

        return self.v

    def _calc_v_lst(self, s):
        factor = 1.0 - s / self.l
        bs = self.lst(self.b_param, s)
        bs_star = self.lst(self.b_warm_param, s)

        numerator = self.p0_star * (factor * bs_star - bs)
        denominator = factor - bs

        if denominator == 0:
            return 1.0

        lst = numerator / denominator

        return lst
