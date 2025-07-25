"""
Class for M/G/1 queue with negative jobs and RCS discipline.
Use following paper:
    Harrison, Peter G., and Edwige Pitel.
    The M/G/1 queue with negative customers
    Advances in Applied Probability 28.2 (1996): 540-566.
"""

import time

from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.transforms import lst_gamma, lst_h2


class MG1NegativeCalcRCS(BaseQueue):
    """
    Class for M/G/1 queue with negative jobs and RCS discipline.
    """

    def __init__(
        self,
        calc_params: CalcParams | None = None,
    ):
        """
        Initialize class.
        :param calc_params: Calculation parameters. If None, default parameters are used.
        """

        super().__init__(n=1, calc_params=calc_params)
        self.l_pos = None
        self.l_neg = None
        self.b = None
        self.approximation = self.calc_params.approx_distr

        self.lst_function = None
        self.params = None

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
        Set the raw moments of service time distribution
        :param b: raw moments of service time distribution
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

    def run(self) -> QueueResults:
        """
        Run calculation
        """
        start = time.process_time()
        v1 = self.get_v1()
        utilization = self.get_utilization()

        return QueueResults(v=[v1, 0.0, 0.0, 0.0], utilization=utilization, duration=time.process_time() - start)

    def get_utilization(self):
        """
        Calculate utilization factor for M/H2/1 queue
        with negative jobs and RCS discipline.
        """
        self._check_if_servers_and_sources_set()
        return self.l_pos * (1 - self.lst_function(self.params, self.l_neg)) / self.l_neg

    def calc_average_jobs_in_system(self) -> float:
        """
        Calculate the average number of jobs in the system for M/H2/1 queue
        with negative jobs and RCS discipline.
        """
        rho = self.get_utilization()
        b_h2_star = self._b_derivative(self.l_neg)

        numerator = self.l_pos * (self.l_pos * b_h2_star + rho)
        denominator = (1 - rho) * self.l_neg

        return rho + numerator / denominator

    def get_v1(self) -> float:
        """
        Calculate the average number of jobs in the system for M/H2/1 queue
        with negative jobs and RCS discipline.
        """
        job_ave = self.calc_average_jobs_in_system()
        return job_ave / self.l_pos

    def _b_derivative(self, s: float) -> float:
        """
        Derivative of b function.
        """
        if self.approximation == "gamma":
            alpha, mu = self.params.alpha, self.params.mu
            return -alpha * (mu**alpha) / ((mu + s) ** (alpha + 1))

        # H2 distribution case
        p1, mu1, mu2 = self.params.p1, self.params.mu1, self.params.mu2
        return -(p1 * (mu1 / ((mu1 + s) ** 2)) + (1.0 - p1) * (mu2 / ((mu2 + s) ** 2)))
