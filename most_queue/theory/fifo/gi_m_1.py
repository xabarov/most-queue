"""
Calculation of the GI/M/1 queueing system
"""

import math
from dataclasses import dataclass

from most_queue.rand_distribution import GammaDistribution, ParetoDistribution
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.conv import conv_moments_minus
from most_queue.theory.utils.q_poisson_arrival_calc import get_q_gamma


@dataclass
class GiM1Result:
    """
    Result of calculation for GI/M/1 system
    """

    v: list[float]  # sojourn time initial moments
    w: list[float]  # waiting time initial moments
    p: list[float]  # probabilities of states
    pi: list[float]  # probabilities of states before arrival
    utilization: float  # utilization factor


class GiM1(BaseQueue):
    """
    Calculation of the GI/M/1 queueing system
    """

    def __init__(self, calc_params: CalcParams | None = None):
        """
        calc_params: calculation parameters
        """

        super().__init__(n=1, calc_params=calc_params)

        self.e = self.calc_params.tolerance
        self.approx_distr = self.calc_params.approx_distr
        self.p_num = self.calc_params.p_num

        self.w_param = None
        self.mu = None
        self.a = None
        self.pi = None

    def set_servers(self, mu: float):  # pylint: disable=arguments-differ
        """
        Setting the service intensity of GI/M/1 queueing system.
        params:
        mu - service intensity
        """
        self.mu = mu
        self.is_servers_set = True

    def set_sources(self, a: list[float]):  # pylint: disable=arguments-differ
        """
        Setting the sources of GI/M/1 queueing system.
        params: a - list of initial moments of arrival distribution.
        """
        self.a = a
        self.is_sources_set = True

    def run(self):
        """
        Run calculation for the GI/M/1 queueing system.
        """
        self._check_if_servers_and_sources_set()

        self.pi = self.get_pi()
        self.p = self.get_p()
        self.w = self.get_w()
        self.v = self.get_v()
        utilization = 1.0 / (self.a[0] * self.mu)

        return GiM1Result(v=self.v, w=self.w, p=self.p, pi=self.pi, utilization=utilization)

    def get_pi(self) -> list[float]:
        """
        Calculation of the probabilities of states before arrival of GI/M/1 queueing system.
        params:
        num - number of states to calculate
        """

        if self.pi:
            return self.pi

        self.w_param = self.w_param or self._get_w_param()

        self.pi = [0.0] * self.p_num

        gamma_params = GammaDistribution.get_params(self.a)

        qs = get_q_gamma(self.mu, gamma_params.mu, gamma_params.alpha)
        summ = 0
        for i, q in enumerate(qs):
            summ += q * pow(self.w_param, i)
        self.pi[0] = 1.0 - summ
        for k in range(1, self.p_num):
            self.pi[k] = (1.0 - self.w_param) * pow(self.w_param, k)
        return self.pi

    def get_v(self, num=3) -> list[float]:
        """
        Calculation of the sojourn time initial moments
        num - number of moments
        e - accuracy
        approx_distr - approximation distribution for the arrival process
        """

        if self.v:
            return self.v

        self.w_param = self.w_param or self._get_w_param()
        v = [0.0] * num
        for k in range(num):
            v[k] = math.factorial(k + 1) / pow(self.mu * (1 - self.w_param), k + 1)

        self.v = v
        return v

    def get_w(self, num=3) -> list[float]:
        """
        Calculation of the initial moments of the waiting time
         num - number of moments
        """

        if self.w:
            return self.w

        self.w_param = self.w_param or self._get_w_param()

        if self.v is None:
            self.v = self.get_v(num)

        b = [
            1.0 / self.mu,
            2.0 / pow(self.mu, 2),
            6.0 / pow(self.mu, 3),
            24.0 / pow(self.mu, 4),
        ]
        self.w = conv_moments_minus(self.v, b, num)

        return self.w

    def get_p(self) -> list[float]:
        """
        Calculation of probabilities of QS states
        num - number of states
        """

        if self.p:
            return self.p

        self.w_param = self.w_param or self._get_w_param()

        ro = 1.0 / (self.a[0] * self.mu)
        p = [0.0] * self.p_num
        p[0] = 1 - ro
        for i in range(1, self.p_num):
            p[i] = ro * (1.0 - self.w_param) * pow(self.w_param, i - 1)

        self.p = p
        return p

    def _get_w_param(self) -> float:
        """
        Calculate w_warm parameter
        """

        self._check_if_servers_and_sources_set()

        ro = 1.0 / (self.a[0] * self.mu)
        coev_a = math.sqrt(self.a[1] - pow(self.a[0], 2)) / self.a[0]
        w_old = pow(ro, 2.0 / (pow(coev_a, 2) + 1.0))

        if self.approx_distr == "gamma":
            gamma_params = GammaDistribution.get_params(self.a)
            while True:
                summ = 0
                for i, q in enumerate(gamma_params.g):
                    numerator = GammaDistribution.get_gamma(gamma_params.alpha + i)
                    denominator = GammaDistribution.get_gamma(gamma_params.alpha)

                    summ += (q / pow(self.mu * (1.0 - w_old) + gamma_params.mu, i)) * (numerator / denominator)
                left = pow(
                    gamma_params.mu / (self.mu * (1.0 - w_old) + gamma_params.mu),
                    gamma_params.alpha,
                )
                w_new = left * summ
                if math.fabs(w_new - w_old) < self.e:
                    break
                w_old = w_new
            return w_new

        if self.approx_distr == "pareto":
            pa_params = ParetoDistribution.get_params(self.a)
            alpha, K = pa_params.alpha, pa_params.K

            while True:
                left = alpha * pow(K * self.mu * (1.0 - w_old), alpha)
                w_new = left * GammaDistribution.get_gamma_incomplete(-alpha, K * self.mu * (1.0 - w_old))
                if math.fabs(w_new - w_old) < self.e:
                    break
                w_old = w_new
            return w_new

        print("w_param calc. Unknown type of distr_type")

        return 0
