"""
Class for calculating M/G/1 queue with warm-up.
"""
import numpy as np
from scipy.misc import derivative

from most_queue.rand_distribution import GammaDistribution, H2Distribution
from most_queue.theory.utils.transforms import lst_gamma, lst_h2


class MG1WarmCalc:
    """
    Class for calculating M/G/1 queue with warm-up.
    """

    def __init__(self, l: float, b: list[float], b_warm: list[float], approximation='gamma'):
        """
        Initialize the MG1WarmCalc class with arrival rate l, 
        service time initial moments b, and warm-up service time moments b_warm.
        Parameters:
        l (float): Arrival rate.
        b (list[float]): Initial moments of service time distribution.
        b_warm (list[float]): Warm-up moments of service time distribution.
        """
        self.l = l
        self.b = b
        self.b_warm = b_warm
        tv = self.b_warm[0] / (1 - self.l * self.b[0])
        self.p0_star = 1 / (1 + self.l * tv)
        self.approximation = approximation
        if approximation == 'gamma':
            self.b_param = GammaDistribution.get_params(self.b)
            self.b_warm_param = GammaDistribution.get_params(self.b_warm)
            self.lst = lst_gamma
        elif approximation == 'h2':
            self.b_param = H2Distribution.get_params(self.b)
            self.b_warm_param = H2Distribution.get_params(self.b_warm)
            self.lst = lst_h2
        else:
            raise ValueError(
                "Invalid approximation method. Must be 'gamma' or 'h2'.")

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

    def get_v(self) -> list[float]:
        """
        Calculate sourjourn moments for M/G/1 queue with warm-up.
        """

        v = [0, 0, 0]

        for i in range(3):
            v[i] = derivative(self._calc_v_lst, 0,
                              dx=1e-3/self.b[0], n=i + 1, order=9)
        return np.array([-v[0], v[1].real, -v[2]])
