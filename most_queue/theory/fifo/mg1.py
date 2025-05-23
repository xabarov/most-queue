"""
Calculation of M/G/1 queue characteristics using the method of moments.
"""
import math

from most_queue.rand_distribution import (
    GammaDistribution,
    ParetoDistribution,
    UniformDistribution,
)
from most_queue.theory.utils.q_poisson_arrival_calc import (
    get_q_gamma,
    get_q_pareto,
    get_q_uniform,
)


class MG1Calculation:
    """
    Calculation of M/G/1 queue characteristics using the method of moments.
    """

    def __init__(self, l: float, b: list[float]):
        """
        :param l: arrival rate
        :param b: initial moments of service time distribution
        """
        self.l = l
        self.b = b

    def get_w(self, num=3) -> list[float]:
        """
        Calculate the initial moments of waiting time for M/G/1 queue.
        """
        num_of_mom = min(len(self.b) - 1, num)
        w = [0.0] * (num_of_mom + 1)
        w[0] = 1
        for k in range(1, num_of_mom + 1):
            summ = 0
            for j in range(k):
                summ += math.factorial(k) * self.b[k - j] * w[j] / (
                    math.factorial(j) * math.factorial(k + 1 - j))
            w[k] = ((self.l / (1 - self.l * self.b[0])) * summ)
        return w[1:]

    def get_v(self, num=3):
        """
        Calculate the initial moments of sojournin the system for M/G/1 queue.
        """
        num_of_mom = min(len(self.b) - 1, num)

        w = self.get_w(num_of_mom)
        v = []
        v.append(w[0] + self.b[0])
        if num_of_mom > 1:
            v.append(w[1] + 2 * w[0] * self.b[0] + self.b[1])
        if num_of_mom > 2:
            v.append(w[2] + 3 * w[1] * self.b[0] +
                     3 * self.b[1] * w[0] + self.b[2])

        return v

    def get_p(self, num=100, dist_type="Gamma"):
        """
        Calculate the probabilities of states for M/G/1 queue.
        num: number of state probabilities to output
        dist_type: type of service time distribution
        """

        if dist_type == "Gamma":
            gamma_param = GammaDistribution.get_params(self.b)
            q = get_q_gamma(self.l, gamma_param.mu, gamma_param.alpha, num)
        elif dist_type == "Uniform":
            uniform_params = UniformDistribution.get_params(self.b)
            q = get_q_uniform(
                self.l, uniform_params.mean, uniform_params.half_interval, num)
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
