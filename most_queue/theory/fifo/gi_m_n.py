"""
Calculation of the GI/M/n queueing system 
"""
import math

import numpy as np

from most_queue.theory.utils.conv import conv_moments
from most_queue.rand_distribution import GammaDistribution, ParetoDistribution
from most_queue.theory.utils.diff5dots import diff5dots


class GiMn:
    """
    Calculation of the GI/M/n queueing system 
    """

    def __init__(self, a: float, mu: float, n: int,  e=1e-10, approx_distr="Gamma", pi_num=100):
        """
        a - list of initial moments of the distribution of inter-renewal intervals of arrival
        mu - service intensity
        n - number of servers in the system
        e - tolerance for convergence
        approx_distr - distribution approximation method, "Gamma" or "Pa" (Pareto)
        pi_num - number of probabilities to calculate
        """
        self.a = a
        self.mu = mu
        self.n = n
        self.e = e
        self.approx_distr = approx_distr
        self.pi_num = pi_num

        self.w_param = self._get_w_param()
        self.pi = self._get_pi()
        self.w = None

    def get_v(self) -> list[float]:
        """
        Calculate sojourn time first 3 initial moments
        """

        if self.w is None:
            w = self.get_w()
        else:
            w = self.w

        b = [1 / self.mu, 2 / pow(self.mu, 2), 6 /
             pow(self.mu, 3), 24 / pow(self.mu, 4)]
        v = conv_moments(w, b, len(w))
        return v

    def get_w(self) -> list[float]:
        """
        Calculate wainig time first 3 initial moments
        """
        pn = self.pi[self.n]
        pls = []
        h = 0.001
        s = 0
        for _ in range(5):
            pls.append(self._get_w_pls(pn, self.w_param, s))
            s += h
        w = diff5dots(pls, h)
        w[0] = - w[0]
        w[2] = - w[2]
        self.w = w
        return w

    def get_p(self) -> list[float]:
        """
        Calculate probabilities of states
        """
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
        return p

    def _get_pi(self):
        """
        Calc pi probabilities using the method of moments.
        """
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
                    A[k, j] += pow(-1, i) * math.factorial(j + 1) * self._get_b0(k + i) / (
                        math.factorial(k) * math.factorial(i) * math.factorial(j + 1 - (k + i)))

            A[k, self.n] = 0
            for i in range(self.n - k + 1):
                A[k, self.n] += pow(-1, i) * math.factorial(self.n) * (self._get_b0(k + i) - self.w_param) / (
                    math.factorial(k) * math.factorial(i) * math.factorial(self.n - k - i) * (self.n * (1 - self.w_param) - (k + i)))
            A[k, self.n] = self.n * A[k, self.n]
            A[k, k] = A[k, k] - 1
        pi_to_n = np.linalg.solve(A, B)
        for i in range(self.n + 1):
            pi[i] = pi_to_n[i]
        for i in range(self.n + 1, self.pi_num):
            pi[i] = pi[self.n] * pow(self.w_param, i - self.n)
        return pi

    def _get_b0(self, j):
        if self.approx_distr == "Gamma":
            gamma_params = GammaDistribution.get_params(self.a)
            v, alpha, gs = gamma_params.mu, gamma_params.alpha, gamma_params.g
            summ = 0
            for i, g_value in enumerate(gs):
                summ += (g_value / pow(self.mu * j + v, i)) * (
                    GammaDistribution.get_gamma(alpha + i) / GammaDistribution.get_gamma(alpha))
            left = pow(v / (self.mu * j + v), alpha)
            b0 = left * summ
            return b0

        elif self.approx_distr == "Pa":
            pa_params = ParetoDistribution.get_params(self.a)
            alpha, K = pa_params.alpha, pa_params.K

            left = alpha * pow(K * self.mu * j, alpha)
            b0 = left * \
                GammaDistribution.get_gamma_incomplete(-alpha, K * self.mu * j)
            return b0

        else:
            print("w_param calc. Unknown type of distr_type")

        return 0

    def _get_w_pls(self, pn, w, s) -> float:
        """
        Calculate Laplace-Stieltjes transform of waiting time
        """
        return self.n * self.mu * pn / (self.n * self.mu * (1.0 - w) + s)

    def _get_w_param(self):
        ro = 1.0 / (self.a[0] * self.mu * self.n)
        coev_a = math.sqrt(self.a[1] - pow(self.a[0], 2)) / self.a[0]
        w_old = pow(ro, 2.0 / (pow(coev_a, 2) + 1.0))

        if self.approx_distr == "Gamma":
            gamma_params = GammaDistribution.get_params(self.a)
            v, alpha, gs = gamma_params.mu, gamma_params.alpha, gamma_params.g

            while True:
                summ = 0
                for i, q_value in enumerate(gs):
                    summ += (q_value / pow(self.mu * self.n * (1.0 - w_old) + v, i)) * (
                        GammaDistribution.get_gamma(alpha + i) / GammaDistribution.get_gamma(alpha))
                left = pow(v / (self.mu * self.n * (1.0 - w_old) + v), alpha)
                w_new = left * summ
                if math.fabs(w_new - w_old) < self.e:
                    break
                w_old = w_new
            return w_new

        elif self.approx_distr == "Pa":
            pa_params = ParetoDistribution.get_params(self.a)
            alpha, K = pa_params.alpha, pa_params.K

            while True:
                left = alpha * pow(K * self.mu * self.n * (1.0 - w_old), alpha)
                w_new = left * \
                    GammaDistribution.get_gamma_incomplete(-alpha,
                                                           K * self.mu * self.n * (1.0 - w_old))
                if math.fabs(w_new - w_old) < self.e:
                    break
                w_old = w_new
            return w_new

        else:
            print("w_param calc. Unknown type of distr_type")

        return 0
