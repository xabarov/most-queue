"""
Calculate distribution of maximum of n independent random variables with given distributions.
"""

import math

from most_queue.random.distributions import (
    ErlangDistribution,
    ErlangParams,
    GammaDistribution,
    GammaParams,
    H2Distribution,
    H2Params,
    ParetoDistribution,
)
from most_queue.random.utils.params import ParetoParams
from most_queue.theory.utils.conv import conv_moments, get_self_conv_moments


def pareto_max_tail(params: ParetoParams, n: int, x: float) -> float:
    """
    Exact P(max(X_1, ..., X_n) > x) for n iid Pareto(alpha, K) random variables.

    ``1 - F(x)^n`` by independence of the max, where F is the Pareto CDF.
    Unlike the moments (see ``pareto_max_moments``), the tail/CDF is always
    well-defined -- valid for any x >= 0, n >= 1, alpha > 0.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    cdf = ParetoDistribution.get_cdf(params, x)
    return 1.0 - cdf**n


def pareto_max_moments(params: ParetoParams, n: int, num: int) -> list[float]:
    """
    Exact raw moments E[max^k], k=1..num, of the maximum of n iid Pareto(alpha, K)
    random variables -- no approximation, no quadrature.

    Derivation: the substitution U = F(X) ~ Uniform(0,1) maps max of n Pareto to
    max of n Uniform(0,1) =: U_(n), with density n*u^(n-1) on (0,1). Then
    ``E[max^k] = K^k * E[(1-U_(n))^(-k/alpha)] = K^k * n * B(n, 1 - k/alpha)``,
    where B is the Beta function -- convergent (as for a single Pareto) only for
    k < alpha.

    Raises ``ValueError`` at the first k >= alpha (moment does not exist),
    rather than the silent truncation ``ParetoDistribution.calc_theory_moments``
    uses: callers here (``MaxDistribution``-style consumers, e.g.
    ``SplitJoinCalc``) expect a fixed-length list, so a clear failure beats a
    silently short one.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    alpha, scale = params.alpha, params.K
    moments = []
    for k in range(1, num + 1):
        if k >= alpha:
            raise ValueError(
                f"E[max^{k}] does not exist for Pareto(alpha={alpha}): requires k < alpha "
                f"(got {len(moments)} of {num} requested moments)"
            )
        # log-space Beta function -- avoids overflow for large n.
        log_beta = math.lgamma(n) + math.lgamma(1.0 - k / alpha) - math.lgamma(n + 1.0 - k / alpha)
        moments.append(math.pow(scale, k) * n * math.exp(log_beta))
    return moments


class MaxDistribution:
    """
    Calculate distribution of maximum of n independent random variables with given distributions.
    """

    def __init__(self, b: list[float], n: int, approximation: str = "gamma"):
        """
        Initialize the MaxDistribution class.
        :param b: List of raw moments of the distributions.
        :param n: Number of distributions.
        :param approximation: approximation of the distribution. Must be 'gamma', 'erlang' or 'h2'
        """
        self.b = b
        self.n = n
        self.approximation = approximation
        self.a = [
            1.37793470540e-1,
            7.29454549503e-1,
            1.808342901740e0,
            3.401433697855e0,
            5.552496140064e0,
            8.330152746764e0,
            1.1843785837900e1,
            1.6279257831378e1,
            2.1996585811981e1,
            2.9920697012274e1,
        ]
        self.g = [
            3.08441115765e-1,
            4.01119929155e-1,
            2.18068287612e-1,
            6.20874560987e-2,
            9.50151697518e-3,
            7.53008388588e-4,
            2.82592334960e-5,
            4.24931398496e-7,
            1.83956482398e-9,
            9.91182721961e-13,
        ]

    def get_max_moments(self):
        """
        Calculate the maximum value of lambda for a given number of channels and service rate.
        The maximum utilization is set to 0.8 by default.
        :param n: number of channels
        :param b: service rate in a channel
        :param num: number of output raw moments of the maximum SV,
        by default one less than the number of raw moments of b
        :return: maximum value of lambda for a given number of channels and service rate.
        """

        if self.approximation == "gamma":
            return self._calc_f_gamma()

        if self.approximation == "h2":
            return self._calc_f_h2()

        return self._calc_f_erlang()

    def get_max_moments_delta(self, delta=0):
        """
        Calculation of the raw moments of the maximum of a random variable with delay delta.
        :param n: number of identically distributed random variables
        :param b: raw moments of the random variable
        :param num: number of raw moments of the random variable
        :return: raw moments of the maximum of the random variable.
        """
        b = self.b

        num = len(self.b)

        f = [0] * num

        if delta:
            params = GammaDistribution.get_params(b)

            for j in range(10):
                p = self.g[j] * self._tail_gamma_mult(params, self.a[j], delta) * math.exp(self.a[j])
                f[0] += p
                for i in range(1, num):
                    p = p * self.a[j]
                    f[i] += p

            for i in range(num - 1):
                f[i + 1] *= i + 2

        return f

    def _calc_f_h2(self):
        num = len(self.b)
        f = [0] * num
        params = H2Distribution.get_params(self.b)

        for j in range(10):
            p = self.g[j] * self._tail_h2_mult(params, self.a[j]) * math.exp(self.a[j])
            f[0] += p
            for i in range(1, num):
                p = p * self.a[j]
                f[i] += p

        for i in range(num - 1):
            f[i + 1] *= i + 2
        return f

    def _calc_f_gamma(self):
        num = len(self.b)
        f = [0] * num
        params = GammaDistribution.get_params(self.b)

        for j in range(10):
            p = self.g[j] * self._tail_gamma_mult(params, self.a[j]) * math.exp(self.a[j])
            f[0] += p
            for i in range(1, num):
                p = p * self.a[j]
                f[i] += p

        for i in range(num - 1):
            f[i + 1] *= i + 2
        return f

    def _calc_f_erlang(self):
        num = len(self.b)

        f = [0] * num

        params = ErlangDistribution.get_params(self.b)

        for j in range(10):
            p = self.g[j] * self._tail_erl_mult(params, self.a[j]) * math.exp(self.a[j])
            f[0] += p
            for i in range(1, num):
                p = p * self.a[j]
                f[i] += p

        for i in range(num - 1):
            f[i + 1] *= i + 2
        return f

    def _tail_h2_mult(self, params: H2Params, t: float, delta=None):
        res = 1.0
        if not delta:
            for i in range(self.n):
                res *= H2Distribution.get_cdf(params, t)
        else:
            if not isinstance(delta, list):
                for i in range(self.n):
                    res *= H2Distribution.get_cdf(params, t - i * delta)
        return 1.0 - res

    def _tail_erl_mult(self, params: ErlangParams, t: float, delta=None):
        res = 1.0
        if not delta:
            for i in range(self.n):
                res *= ErlangDistribution.get_cdf(params, t)
        else:
            if not isinstance(delta, list):
                for i in range(self.n):
                    res *= ErlangDistribution.get_cdf(params, t - i * delta)
        return 1.0 - res

    def _tail_gamma_mult(self, params: GammaParams, t: float, delta=None):
        res = 1.0
        if not delta:
            for i in range(self.n):
                res *= GammaDistribution.get_cdf(params, t)
        else:
            if not isinstance(delta, list):
                for i in range(self.n):
                    res *= GammaDistribution.get_cdf(params, t - i * delta)
            else:
                b = GammaDistribution.calc_theory_moments(params)

                for i in range(self.n):
                    b_delta = get_self_conv_moments(delta, i)
                    b_summ = conv_moments(b, b_delta)
                    params_summ = GammaDistribution.get_params(b_summ)
                    res *= GammaDistribution.get_cdf(params_summ, t)

        return 1.0 - res
