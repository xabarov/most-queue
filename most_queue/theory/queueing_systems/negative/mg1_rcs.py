"""
Class for M/G/1 queue with negative jobs and RCS discipline.
Use following paper: 
    Harrison, Peter G., and Edwige Pitel. 
    \"The M/G/1 queue with negative customers.\"
    Advances in Applied Probability 28.2 (1996): 540-566.
"""
from abc import ABC, abstractmethod

from most_queue.rand_distribution import GammaParams, H2Params
from most_queue.theory.utils.transforms import lst_gamma, lst_h2


class MG1_RCS(ABC):
    """
    Class for M/G/1 queue with negative jobs and RCS discipline.
    """

    def __init__(self, l_pos: float, l_neg: float, params, lst_function):
        self.l_pos = l_pos
        self.l_neg = l_neg
        self.params = params
        self.lst_function = lst_function

    @abstractmethod
    def b_derivative(self, s: float) -> float:
        """
        Derivative of b function.
        """

    def calc_rho(self):
        """
        Calculate utilization factor for M/H2/1 queue 
        with negative jobs and RCS discipline.
        """
        return self.l_pos*(1-self.lst_function(self.params, self.l_neg))/self.l_neg

    def calc_average_jobs_in_system(self) -> float:
        """
        Calculate the average number of jobs in the system for M/H2/1 queue
        with negative jobs and RCS discipline.
        """
        rho = self.calc_rho()
        b_h2_star = self.b_derivative(self.l_neg)

        numerator = self.l_pos*(self.l_pos*b_h2_star + rho)
        denominator = (1 - rho)*self.l_neg

        return rho + numerator / denominator

    def get_v1(self) -> float:
        """
        Calculate the average number of jobs in the system for M/H2/1 queue
        with negative jobs and RCS discipline.
        """
        job_ave = self.calc_average_jobs_in_system()
        return job_ave/(self.l_pos)


class M_H2_1_RCS(MG1_RCS):
    """
    Class for M/H2/1 queue with negative jobs and RCS discipline.
    """

    def __init__(self, l_pos: float, l_neg: float, params: H2Params):

        super().__init__(l_pos, l_neg, params=params, lst_function=lst_h2)

    def b_derivative(self, s: float) -> float:
        """
        Calculate the derivative of the Laplace-Stieltjes transform 
        of an H2 distribution.
        """
        p1, mu1, mu2 = self.params.p1, self.params.mu1, self.params.mu2
        return - (p1 * (mu1 / ((mu1 + s)**2)) + (1.0 - p1) * (mu2 / ((mu2 + s)**2)))


class M_Gamma_1_RCS(MG1_RCS):
    """
    Class for M/Gamma/1 queue with negative jobs and RCS discipline.

    """

    def __init__(self, l_pos: float, l_neg: float, params: GammaParams):
        super().__init__(l_pos, l_neg, params=params, lst_function=lst_gamma)

    def b_derivative(self, s: float) -> float:
        """
        Calculate the derivative of the Laplace-Stieltjes transform 
        of an Gamma distribution.
        """
        alpha, mu = self.params.alpha, self.params.mu
        return -alpha * (mu ** alpha) / ((mu + s) ** (alpha + 1))
