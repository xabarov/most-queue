"""
Calculation of the GI/M/1 queueing system 
"""
import math

from most_queue.rand_distribution import GammaDistribution, ParetoDistribution
from most_queue.theory.utils.conv import conv_moments_minus
from most_queue.theory.utils.q_poisson_arrival_calc import get_q_gamma


class GiM1:
    """
    Calculation of the GI/M/1 queueing system 
    """

    def __init__(self, a: list[float], mu: float, tolerance: float = 1e-10, approx_distr="Gamma"):
        """
        a - list of initial moments of the distribution of inter-renewal intervals of arrival
        mu - service intensity
        approx_distr - distribution approximation method, "Gamma" or "Pa" (Pareto)
        """
        self.a = a
        self.mu = mu
        self.e = tolerance
        self.approx_distr = approx_distr

        self.w_param = self._get_w_param()

        self.v = None
        self.w = None
        self.pi = None

    def get_pi(self, num=100):
        """
        Calculation of the probabilities of states before arrival of GI/M/1 queueing system.
        params:
        num - number of states to calculate
        """

        pi = [0.0] * num

        gamma_params = GammaDistribution.get_params(self.a)

        qs = get_q_gamma(self.mu, gamma_params.mu, gamma_params.alpha)
        summ = 0
        for i, q in enumerate(qs):
            summ += q * pow(self.w_param, i)
        pi[0] = 1.0 - summ
        for k in range(1, num):
            pi[k] = (1.0 - self.w_param) * pow(self.w_param, k)
        return pi

    def get_v(self, num=3):
        """
        Calculation of the sojourn time initial moments
        num - number of moments
        e - accuracy
        approx_distr - approximation distribution for the arrival process
        """
        v = [0.0] * num
        for k in range(num):
            v[k] = math.factorial(k + 1) / pow(self.mu *
                                               (1 - self.w_param), k + 1)
        return v

    def get_w(self, num=3):
        """
        Calculation of the initial moments of the waiting time
         num - number of moments
        """

        if self.v is None:
            self.v = self.get_v(num)

        b = [1.0 / self.mu, 2.0 /
             pow(self.mu, 2), 6.0 / pow(self.mu, 3), 24.0 / pow(self.mu, 4)]
        w = conv_moments_minus(self.v, b, num)

        return w

    def get_p(self, num=100):
        """
        Calculation of probabilities of QS states
        num - number of states
        """
        ro = 1.0 / (self.a[0] * self.mu)
        p = [0.0] * num
        p[0] = 1 - ro
        for i in range(1, num):
            p[i] = ro * (1.0 - self.w_param) * pow(self.w_param, i - 1)
        return p

    def _get_w_param(self) -> float:
        """
        Calculate w_warm parameter
        """
        ro = 1.0 / (self.a[0] * self.mu)
        coev_a = math.sqrt(self.a[1] - pow(self.a[0], 2)) / self.a[0]
        w_old = pow(ro, 2.0 / (pow(coev_a, 2) + 1.0))

        if self.approx_distr == "Gamma":
            gamma_params = GammaDistribution.get_params(self.a)
            while True:
                summ = 0
                for i, q in enumerate(gamma_params.g):
                    summ += (q / pow(self.mu * (1.0 - w_old) + gamma_params.mu, i)) * (
                        GammaDistribution.get_gamma(gamma_params.alpha + i) / GammaDistribution.get_gamma(gamma_params.alpha))
                left = pow(gamma_params.mu / (self.mu *
                           (1.0 - w_old) + gamma_params.mu), gamma_params.alpha)
                w_new = left * summ
                if math.fabs(w_new - w_old) < self.e:
                    break
                w_old = w_new
            return w_new

        elif self.approx_distr == "Pa":
            pa_params = ParetoDistribution.get_params(self.a)
            alpha, K = pa_params.alpha, pa_params.K

            while True:
                left = alpha * pow(K * self.mu * (1.0 - w_old), alpha)
                w_new = left * \
                    GammaDistribution.get_gamma_incomplete(-alpha,
                                                           K * self.mu * (1.0 - w_old))
                if math.fabs(w_new - w_old) < self.e:
                    break
                w_old = w_new
            return w_new

        else:
            print("w_param calc. Unknown type of distr_type")

        return 0
