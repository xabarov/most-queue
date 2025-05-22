"""
Class for calculating M/G/1 queue with disasters.
Use results from the paper:
    Jain, Gautam, and Karl Sigman. "A Pollaczek–Khintchine formula for M/G/1 queues with disasters."
    Journal of Applied Probability 33.4 (1996): 1191-1200.
"""
import numpy as np
from scipy.misc import derivative

from most_queue.rand_distribution import H2Distribution
from most_queue.theory.utils.busy_periods import busy_calc, busy_calc_lst
from most_queue.theory.utils.transforms import lst_h2


class MG1Disasters:
    """
    Class for calculating M/G/1 queue with disasters.
    """

    def __init__(self, l_pos: float, l_neg: float, b: list[float]):
        """
        Parameters
        ----------
        l_pos : float
            Arrival rate of positive customers.
        l_neg : float
           Arrival rate of negative customers.
        b : list[float]
        Arrival rates of disasters.
        """
        self.l_pos = l_pos
        self.l_neg = l_neg
        self.b = b
        self.lst_function = lst_h2
        self.params = H2Distribution.get_params(b)

        self.nu = self._calc_nu()

    def _calc_nu(self):
        """
        Calculate the nu parameter.
        Notice: l_neg*E[min{B, Y}] = 1-b_lst(l_neg)),
        so nu = E[min{B, Y}]/(1/l_neg + E[min{B, Y}]) = 
        (1-b_lst(l_neg))/(self.l_neg/self.l_pos + 1 - b_lst(l_neg))
        """
        busy_moments = busy_calc(self.l_pos, self.b)
        busy_params = H2Distribution.get_params(busy_moments)
        one_minus_b_lst = 1.0 - self.lst_function(busy_params, self.l_neg)
        nu = one_minus_b_lst/(self.l_neg/self.l_pos + one_minus_b_lst)
        return nu

    def _v_lst(self, s):
        """
        Calculate  Laplace-Stieljets transform for sojourn time in the system.
        """
        numerator = s*(1-self.nu) - self.l_neg
        big_g = self.lst_function(self.params, s)
        denominator = s - self.l_pos * (1.0-big_g) - self.l_neg
        return numerator/denominator

    def get_v(self) -> list[float]:
        """
        Calculate first three moments of sojourn time in the system.
        """
        v = [0, 0, 0]
        for i in range(3):
            v[i] = derivative(self._v_lst, 0,
                              dx=1e-3 / self.b[0], n=i + 1, order=9)
        return np.array([-v[0], v[1].real, -v[2]])
