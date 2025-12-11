"""
Calculation of M/G/1 queue characteristics using the method of moments.
"""

import math

from most_queue.random.distributions import GammaDistribution, ParetoDistribution, UniformDistribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.q_poisson_arrival_calc import get_q_gamma, get_q_pareto, get_q_uniform


class MG1Calc(BaseQueue):
    """
    Calculation of M/G/1 queue characteristics using the method of moments.
    """

    def __init__(self, calc_params: CalcParams | None = None) -> None:
        """
        Initialize the MG1Calc class.

        Args:
            calc_params: Calculation parameters. If None, default CalcParams will be used.
        """
        super().__init__(n=1, calc_params=calc_params)

        self.l = None
        self.b = None

    def set_sources(self, l: float) -> None:  # pylint: disable=arguments-differ
        """
        Set the arrival rate.

        Args:
            l: Arrival rate (lambda).
        """
        self.l = l
        self.is_sources_set = True

    def set_servers(self, b: list[float]) -> None:  # pylint: disable=arguments-differ
        """
        Set the raw moments of service time distribution.

        Args:
            b: Raw moments of service time distribution. b[0] is the mean, b[1] is the second moment, etc.
        """
        self.b = b
        self.is_servers_set = True

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """
        Run calculation for M/G/1 queue.

        Args:
            num_of_moments: Number of moments to calculate.

        Returns:
            QueueResults with calculated values.
        """
        start = self._measure_time()

        w = self.get_w(num_of_moments)
        v = self.get_v(num_of_moments)
        p = self.get_p()

        result = QueueResults(v=v, w=w, p=p, utilization=self.l * self.b[0])
        self._set_duration(result, start)
        return result

    def get_w(self, num: int = 3) -> list[float]:
        """
        Calculate the raw moments of waiting time for M/G/1 queue.
        """

        self._check_if_servers_and_sources_set()

        if self.w:
            return self.w

        num_of_mom = min(len(self.b) - 1, num)

        w = [0.0] * (num_of_mom + 1)
        w[0] = 1
        for k in range(1, num_of_mom + 1):
            summ = 0
            for j in range(k):
                summ += math.factorial(k) * self.b[k - j] * w[j] / (math.factorial(j) * math.factorial(k + 1 - j))
            w[k] = (self.l / (1 - self.l * self.b[0])) * summ

        self.w = w[1:]
        return self.w

    def get_v(self, num: int = 3) -> list[float]:
        """
        Calculate the raw moments of sojournin the system for M/G/1 queue.
        """

        self._check_if_servers_and_sources_set()

        if self.v:
            return self.v

        num_of_mom = min(len(self.b) - 1, num)

        w = self.w or self.get_w(num_of_mom)
        self.v = conv_moments(w, self.b, num=num_of_mom)

        return self.v

    def get_p(self) -> list[float]:
        """
        Calculate the probabilities of states for M/G/1 queue.
        num: number of state probabilities to output
        dist_type: type of service time distribution
        """

        self._check_if_servers_and_sources_set()

        if self.p:
            return self.p

        if self.calc_params.approx_distr == "gamma":
            gamma_param = GammaDistribution.get_params(self.b)
            q = get_q_gamma(self.l, gamma_param.mu, gamma_param.alpha, self.calc_params.p_num)
        elif self.calc_params.approx_distr == "uniform":
            uniform_params = UniformDistribution.get_params(self.b)
            q = get_q_uniform(self.l, uniform_params.mean, uniform_params.half_interval, self.calc_params.p_num)
        elif self.calc_params.approx_distr == "pa":
            pa_params = ParetoDistribution.get_params(self.b)
            q = get_q_pareto(self.l, pa_params.alpha, pa_params.K, self.calc_params.p_num)
        else:
            raise ValueError(f"Unknown type of distribution: {self.calc_params.approx_distr}")

        p = [0.0] * self.calc_params.p_num
        p[0] = 1 - self.l * self.b[0]
        for i in range(1, self.calc_params.p_num):
            summ = 0
            for j in range(1, i):
                summ += p[j] * q[i - j]
            p[i] = (p[i - 1] - p[0] * q[i - 1] - summ) / q[0]

        self.p = p
        return p
