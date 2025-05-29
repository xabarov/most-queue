"""
Class for M/G/1 queue with negative jobs and RCS discipline.
Use following paper: 
    Harrison, Peter G., and Edwige Pitel. 
    The M/G/1 queue with negative customers
    Advances in Applied Probability 28.2 (1996): 540-566.
"""

from most_queue.rand_distribution import GammaDistribution, H2Distribution
from most_queue.theory.utils.transforms import lst_gamma, lst_h2


class MG1NegativeCalcRCS:
    """
    Class for M/G/1 queue with negative jobs and RCS discipline.
    """

    def __init__(self, l_pos: float, l_neg: float, b: list[float],
                 service_time_approx_dist='gamma'):
        """
        Parameters
        ----------
        l_pos : float
            Arrival rate of positive customers.
        l_neg : float
            Arrival rate of negative customers.
        b : list[float]
            Service time moments.
        service_time_approx_dist : str
            Service time approximation distribution. Can be 'gamma' or 'h2'.
        """
        self.l_pos = l_pos
        self.l_neg = l_neg
        self.b = b
        self.service_time_approx_dist = service_time_approx_dist
        if service_time_approx_dist == 'gamma':
            self.lst_function = lst_gamma
            self.params = GammaDistribution.get_params(b)
        elif service_time_approx_dist == 'h2':
            self.lst_function = lst_h2
            self.params = H2Distribution.get_params(b)
        else:
            raise ValueError(
                "Invalid service time approximation distribution. Must be 'gamma' or 'h2'.")

    def b_derivative(self, s: float) -> float:
        """
        Derivative of b function.
        """
        if self.service_time_approx_dist == 'gamma':
            alpha, mu = self.params.alpha, self.params.mu
            return -alpha * (mu ** alpha) / ((mu + s) ** (alpha + 1))

        # H2 distribution case
        p1, mu1, mu2 = self.params.p1, self.params.mu1, self.params.mu2
        return - (p1 * (mu1 / ((mu1 + s)**2)) + (1.0 - p1) * (mu2 / ((mu2 + s)**2)))

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
        return job_ave/self.l_pos
