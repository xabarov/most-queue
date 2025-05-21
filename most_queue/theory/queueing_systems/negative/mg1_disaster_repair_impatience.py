"""
Class for M/G/1 queue with disasters, repairs and impatience customers when system is down.
Use following paper: 
    M/G/1 QUEUE WITH SYSTEM DISASTERS AND IMPATIENT
    CUSTOMERS WHEN SYSTEM IS DOWN
    c Kovalenko A. I., Smolich V. P.
"""

import numpy as np
from scipy.integrate import quad

from most_queue.rand_distribution import GammaDistribution, H2Distribution
from most_queue.theory.utils.transforms import lst_gamma, lst_h2


def a_func(y, z, repair_params, l_pos, alpha):
    f2 = H2Distribution.get_pdf(repair_params, y)
    x = -l_pos*(1-z)*(1-np.exp(-alpha*y))/alpha
    return f2*np.exp(x)


class MG1DisasterRepairImpatienceCalc:
    """
    Class for M/G/1 queue with disasters, repairs and impatience customers when system is down.
    """

    def __init__(self, l_pos: float, l_neg: float, b: list[float], repair_moments: list[float],
                 impatience_rate: float,
                 approx_dist='gamma'):
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
        repair_moments : list[float]
            Repair moments.
        impatience_rate : float
            Impatience rate of customers when system is down.
        repair_time_approx_dist : str
            Repair time approximation distribution. Can be 'gamma' or 'h2'.
        """
        self.l_pos = l_pos
        self.l_neg = l_neg
        self.b = b
        self.repair_moments = repair_moments
        self.impatience_rate = impatience_rate

        self.approx_dist = approx_dist
        if approx_dist == 'gamma':
            self.lst_function = lst_gamma
            self.params = GammaDistribution.get_params(b)
            self.repair_params = GammaDistribution.get_params(repair_moments)
        elif approx_dist == 'h2':
            self.lst_function = lst_h2
            self.params = H2Distribution.get_params(b)
            self.repair_params = H2Distribution.get_params(repair_moments)
        else:
            raise ValueError(
                "Invalid service time approximation distribution. Must be 'gamma' or 'h2'.")

    def calc_v1_in_down_state(self):
        """
        Calculate the average sojourn time for a job in down state.
        """
        return self.calc_ave_jobs_in_down_state()/self.l_pos

    def calc_v1_in_up_state(self):
        """
        Calculate the average sojourn time for a job in up state.
        """
        return self.calc_ave_jobs_in_up_state()/self.l_pos

    def calc_ave_jobs_in_down_state(self):
        """
        Calculate the average number of jobs in the down state.
        """
        p0dot = self._calc_p0_dot()
        big_phi2 = self._calc_big_phi2(self.impatience_rate)
        return p0dot * self.l_pos*(1-big_phi2/self.repair_moments[0])/self.impatience_rate

    def calc_ave_jobs_in_up_state(self):
        """
        Calculate the average number of jobs in the up state.
        """
        p10 = self._calc_p10()
        p1dot = self._calc_p1_dot()
        big_phi2 = self._calc_big_phi2(self.impatience_rate)
        f1_lst = lst_h2(self.params, self.l_neg)

        first = f1_lst*(p10-p1dot)/(1-f1_lst)
        second = self.l_pos*p1dot*(1/self.impatience_rate + big_phi2)
        return first + second

    def _calc_big_phi2(self, s):

        return (1 - lst_h2(self.repair_params, s))/s

    def _calc_p0_dot(self) -> float:

        return self.l_neg*self.repair_moments[0]/(1 + self.l_neg*self.repair_moments[0])

    def _calc_p1_dot(self) -> float:

        return 1.0/(1 + self.l_neg*self.repair_moments[0])

    def _find_z0(self, tolerance: float = 1e-6) -> float:
        """
        Find the value of z0 for a given z.
        """
        z = 0.5
        while True:
            s = self.l_pos*(1-z) + self.l_neg
            z_new = lst_h2(self.params, s)
            if abs(z - z_new) < tolerance:
                return z_new
            z = z_new

    def _calc_p10(self) -> float:
        z0 = self._find_z0()
        az0 = self._find_big_a(z0)
        p1_dot = self._calc_p1_dot()
        return self.l_neg*az0*p1_dot/(self.l_pos*(1 - z0) + self.l_neg)

    def _find_big_a(self, z):
        args = (z, self.repair_params, self.l_pos, self.impatience_rate)
        big_a, _err = quad(a_func, 0, 1e3, args=args, limit=100)
        return big_a
