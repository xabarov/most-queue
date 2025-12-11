"""
Calculation of the GI/M/1 queueing system
"""

import math

from most_queue.random.distributions import GammaDistribution, ParetoDistribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.conv import conv_moments_minus
from most_queue.theory.utils.q_poisson_arrival_calc import get_q_gamma


class GIM1Calc(BaseQueue):
    """
    Calculation of the GI/M/1 queueing system.
    """

    def __init__(self, calc_params: CalcParams | None = None) -> None:
        """
        Initialize the GIM1Calc class.

        Args:
            calc_params: Calculation parameters. If None, default CalcParams will be used.
        """

        super().__init__(n=1, calc_params=calc_params)

        self.e = self.calc_params.tolerance
        self.approx_distr = self.calc_params.approx_distr
        self.p_num = self.calc_params.p_num

        self.w_param = None
        self.mu = None
        self.a = None
        self.pi = None

    def set_servers(self, mu: float) -> None:  # pylint: disable=arguments-differ
        """
        Set the service intensity of GI/M/1 queueing system.

        Args:
            mu: Service intensity (service rate).
        """
        self.mu = mu
        self.is_servers_set = True

    def set_sources(self, a: list[float]) -> None:  # pylint: disable=arguments-differ
        """
        Set the sources of GI/M/1 queueing system.

        Args:
            a: List of raw moments of arrival distribution. a[0] is the mean, a[1] is the second moment, etc.
        """
        self.a = a
        self.is_sources_set = True

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """
        Run calculation for the GI/M/1 queueing system.

        Args:
            num_of_moments: Number of moments to calculate.

        Returns:
            QueueResults with calculated values.
        """
        start = self._measure_time()

        self._check_if_servers_and_sources_set()

        self.p = self.get_p()
        self.w = self.get_w(num_of_moments)
        self.v = self.get_v(num_of_moments)
        utilization = 1.0 / (self.a[0] * self.mu)

        result = QueueResults(v=self.v, w=self.w, p=self.p, pi=self.pi, utilization=utilization)
        self._set_duration(result, start)
        return result

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

    def get_v(self, num: int = 3) -> list[float]:
        """
        Calculation of the sojourn time raw moments
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

    def get_w(self, num: int = 3) -> list[float]:
        """
        Calculation of the raw moments of the waiting time
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
        cv_a = math.sqrt(self.a[1] - pow(self.a[0], 2)) / self.a[0]
        w_old = pow(ro, 2.0 / (pow(cv_a, 2) + 1.0))

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

        raise ValueError(f"Unknown type of distribution: {self.approx_distr}")
