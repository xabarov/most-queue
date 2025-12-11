"""
Calculation of the GI/M/n queueing system
"""

import math
import time

import numpy as np
from scipy.misc import derivative

from most_queue.random.distributions import GammaDistribution, ParetoDistribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.diff5dots import diff5dots


class GiMn(BaseQueue):
    """
    Calculation of the GI/M/n queueing system
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        """
        n - number of servers in the system
        calc_params - calculation parameters
        """

        super().__init__(n=n, calc_params=calc_params)

        self.e = self.calc_params.tolerance
        self.approx_distr = self.calc_params.approx_distr
        self.pi_num = self.calc_params.p_num

        self.w_param = None
        self.pi = None
        self.mu = None
        self.a = None

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
        params: a - list of raw moments of arrival distribution.
        """
        self.a = a
        self.is_sources_set = True

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """
        Run calculation for the GI/M/1 queueing system.
        """

        start = time.process_time()

        self._check_if_servers_and_sources_set()

        self.p = self.get_p()
        self.w = self.get_w(num_of_moments)
        self.v = self.get_v(num_of_moments)
        utilization = 1.0 / (self.a[0] * self.mu * self.n)

        return QueueResults(
            v=self.v, w=self.w, p=self.p, pi=self.pi, utilization=utilization, duration=time.process_time() - start
        )

    def get_v(self, num: int = 4) -> list[float]:
        """
        Calculate sojourn time first 3 raw moments
        """

        if self.v:
            return self.v

        if self.w is None:
            self.w = self.get_w(num)

        b = [
            1 / self.mu,
            2 / pow(self.mu, 2),
            6 / pow(self.mu, 3),
            24 / pow(self.mu, 4),
        ]
        self.v = conv_moments(self.w, b, num)
        return self.v

    def get_w(self, num: int = 4) -> list[float]:
        """
        Calculate wainig time first 3 raw moments
        """

        if self.w:
            return self.w

        self.w_param = self.w_param or self._get_w_param()
        self.pi = self.get_pi() or self.pi

        w = [0.0] * num

        for i in range(num):
            w[i] = derivative(self._get_w_pls, 0, dx=1e-4, n=i + 1, order=9)
            if i % 2 == 0:
                w[i] = -w[i]
        self.w = w
        return w

    def get_p(self) -> list[float]:
        """
        Calculate probabilities of states
        """

        if self.p:
            return self.p

        self.w_param = self.w_param or self._get_w_param()
        self.pi = self.get_pi() or self.pi

        p = [0.0] * self.pi_num
        for i in range(1, self.n + 1):
            p[i] = self.pi[i - 1] / (i * self.mu * self.a[0])
        for i in range(self.n, self.pi_num):
            p[i] = self.pi[i - 1] / (self.n * self.mu * self.a[0])
        summ = 0
        for k in range(1, self.n):
            summ += (self.n - k) * p[k]
        summ = summ + 1.0 / (self.mu * self.a[0])
        p[0] = 1.0 - summ / self.n
        self.p = p
        return p

    def get_pi(self):
        """
        Calc pi probabilities using the method of moments.
        """

        if self.pi:
            return self.pi

        self.w_param = self.w_param or self._get_w_param()

        pi = [0.0] * self.pi_num
        A = np.zeros((self.n + 1, self.n + 1))
        B = np.zeros(self.n + 1)

        B[0] = 1
        for i in range(self.n):
            A[0, i] = 1

        A[0, self.n] = 1.0 / (1.0 - self.w_param)

        for k in range(1, self.n + 1):
            for j in range(k - 1, self.n):
                A[k, j] = 0
                for i in range(j + 2 - k):
                    A[k, j] += (
                        pow(-1, i)
                        * math.factorial(j + 1)
                        * self._get_b0(k + i)
                        / (math.factorial(k) * math.factorial(i) * math.factorial(j + 1 - (k + i)))
                    )

            A[k, self.n] = 0
            for i in range(self.n - k + 1):
                val2 = math.factorial(self.n)
                val3 = math.factorial(k)
                val4 = math.factorial(i)
                val5 = math.factorial(self.n - k - i)
                val6 = self.n * (1 - self.w_param) - (k + i)
                A[k, self.n] += pow(-1, i) * val2 * (self._get_b0(k + i) - self.w_param) / (val3 * val4 * val5 * val6)
            A[k, self.n] = self.n * A[k, self.n]
            A[k, k] = A[k, k] - 1
        pi_to_n = np.linalg.solve(A, B)
        for i in range(self.n + 1):
            pi[i] = pi_to_n[i]
        for i in range(self.n + 1, self.pi_num):
            pi[i] = pi[self.n] * pow(self.w_param, i - self.n)

        self.pi = pi
        return pi

    def _get_b0(self, j):

        if self.approx_distr == "gamma":
            gamma_params = GammaDistribution.get_params(self.a)
            v, alpha, gs = gamma_params.mu, gamma_params.alpha, gamma_params.g
            summ = 0
            for i, g_value in enumerate(gs):
                summ += (g_value / pow(self.mu * j + v, i)) * (
                    GammaDistribution.get_gamma(alpha + i) / GammaDistribution.get_gamma(alpha)
                )
            left = pow(v / (self.mu * j + v), alpha)
            b0 = left * summ
            return b0

        if self.approx_distr == "pareto":
            pa_params = ParetoDistribution.get_params(self.a)
            alpha, K = pa_params.alpha, pa_params.K

            left = alpha * pow(K * self.mu * j, alpha)
            b0 = left * GammaDistribution.get_gamma_incomplete(-alpha, K * self.mu * j)
            return b0

        raise ValueError(f"Unknown type of distribution: {self.approx_distr}")

    def _get_w_pls(self, s) -> float:
        """
        Calculate Laplace-Stieltjes transform of waiting time
        """
        pn = self.pi[self.n]
        return self.n * self.mu * pn / (self.n * self.mu * (1.0 - self.w_param) + s)

    def _get_w_param(self):

        self._check_if_servers_and_sources_set()

        ro = 1.0 / (self.a[0] * self.mu * self.n)
        cv_a = math.sqrt(self.a[1] - pow(self.a[0], 2)) / self.a[0]
        w_old = pow(ro, 2.0 / (pow(cv_a, 2) + 1.0))

        if self.approx_distr == "gamma":
            gamma_params = GammaDistribution.get_params(self.a)
            v, alpha, gs = gamma_params.mu, gamma_params.alpha, gamma_params.g

            while True:
                summ = 0
                for i, q_value in enumerate(gs):
                    summ += (q_value / pow(self.mu * self.n * (1.0 - w_old) + v, i)) * (
                        GammaDistribution.get_gamma(alpha + i) / GammaDistribution.get_gamma(alpha)
                    )
                left = pow(v / (self.mu * self.n * (1.0 - w_old) + v), alpha)
                w_new = left * summ
                if math.fabs(w_new - w_old) < self.e:
                    break
                w_old = w_new
            return w_new

        if self.approx_distr == "pareto":
            pa_params = ParetoDistribution.get_params(self.a)
            alpha, K = pa_params.alpha, pa_params.K

            while True:
                left = alpha * pow(K * self.mu * self.n * (1.0 - w_old), alpha)
                w_new = left * GammaDistribution.get_gamma_incomplete(-alpha, K * self.mu * self.n * (1.0 - w_old))
                if math.fabs(w_new - w_old) < self.e:
                    break
                w_old = w_new
            return w_new

        raise ValueError(f"Unknown type of distribution: {self.approx_distr}")
