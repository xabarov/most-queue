"""
Calculation of M/G/1 queue characteristics using the method of moments.
"""

import math

from most_queue.rand_distribution import GammaDistribution, ParetoDistribution, UniformDistribution
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.q_poisson_arrival_calc import get_q_gamma, get_q_pareto, get_q_uniform


class MG1Calculation(BaseQueue):
    """
    Calculation of M/G/1 queue characteristics using the method of moments.
    """

    def __init__(self):
        """
        Initialize the MG1Calculation class.
        """
        super().__init__(n=1)

        self.l = None
        self.b = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set the arrival rate.
        """
        self.l = l
        self.is_sources_set = True

    def set_servers(self, b: list[float]):  # pylint: disable=arguments-differ
        """
        Set the initial moments of service time distribution.
        param b: initial moments of service time distribution.
        """
        self.b = b
        self.is_servers_set = True

    def get_w(self, num=3) -> list[float]:
        """
        Calculate the initial moments of waiting time for M/G/1 queue.
        """

        self._check_if_servers_and_sources_set()

        num_of_mom = min(len(self.b) - 1, num)

        w = [0.0] * (num_of_mom + 1)
        w[0] = 1
        for k in range(1, num_of_mom + 1):
            summ = 0
            for j in range(k):
                summ += math.factorial(k) * self.b[k - j] * w[j] / (math.factorial(j) * math.factorial(k + 1 - j))
            w[k] = (self.l / (1 - self.l * self.b[0])) * summ
        return w[1:]

    def get_v(self, num=3):
        """
        Calculate the initial moments of sojournin the system for M/G/1 queue.
        """

        self._check_if_servers_and_sources_set()

        num_of_mom = min(len(self.b) - 1, num)

        w = self.get_w(num_of_mom)
        v = conv_moments(w, self.b)

        return v

    def get_p(self, num=100, dist_type="Gamma"):
        """
        Calculate the probabilities of states for M/G/1 queue.
        num: number of state probabilities to output
        dist_type: type of service time distribution
        """

        self._check_if_servers_and_sources_set()

        if dist_type == "Gamma":
            gamma_param = GammaDistribution.get_params(self.b)
            q = get_q_gamma(self.l, gamma_param.mu, gamma_param.alpha, num)
        elif dist_type == "Uniform":
            uniform_params = UniformDistribution.get_params(self.b)
            q = get_q_uniform(self.l, uniform_params.mean, uniform_params.half_interval, num)
        elif dist_type == "Pa":
            pa_params = ParetoDistribution.get_params(self.b)
            q = get_q_pareto(self.l, pa_params.alpha, pa_params.K, num)
        else:
            print("Error in get_p. Unknown type of distribution")
            return 0

        p = [0.0] * num
        p[0] = 1 - self.l * self.b[0]
        for i in range(1, num):
            summ = 0
            for j in range(1, i):
                summ += p[j] * q[i - j]
            p[i] = (p[i - 1] - p[0] * q[i - 1] - summ) / q[0]
        return p
