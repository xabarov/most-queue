"""
Calculate distribution of maximum of n independent random variables with given distributions.
"""

import math
from typing import Any, Callable

from scipy.integrate import quad

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

BranchSpec = tuple[str, Any]  # (family, ParetoParams | raw moments), see branch_tail()


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


def pareto_kth_order_moments(params: ParetoParams, n: int, k: int, num: int) -> list[float]:
    """
    Exact raw moments E[X_(k)^m], m=1..num, of the k-th order statistic
    (ascending: k=1 the minimum, k=n the maximum) of n iid Pareto(alpha, K)
    random variables -- no approximation, no quadrature. Generalizes
    ``pareto_max_moments`` (the k=n special case, exact regression check).

    Derivation: as in ``pareto_max_moments``, substitute U = F(X) ~
    Uniform(0,1); X_(k) = K / (1 - U_(k))^(1/alpha), and the k-th order
    statistic of n Uniform(0,1) is U_(k) ~ Beta(k, n-k+1), so
    1 - U_(k) ~ Beta(n-k+1, k). Then
    ``E[X_(k)^m] = K^m * E[(1-U_(k))^(-m/alpha)]
                 = K^m * B(n-k+1 - m/alpha, k) / B(n-k+1, k)``,
    convergent only for m < alpha*(n-k+1) (reduces to the existing m < alpha
    condition at k=n).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")
    alpha, scale = params.alpha, params.K
    moments = []
    for m in range(1, num + 1):
        if m >= alpha * (n - k + 1):
            raise ValueError(
                f"E[X_({k})^{m}] does not exist for Pareto(alpha={alpha}), n={n}: requires "
                f"m < alpha*(n-k+1) (got {len(moments)} of {num} requested moments)"
            )
        # log-space Beta-function ratio -- avoids overflow, see docstring.
        log_ratio = (
            math.lgamma(n - k + 1 - m / alpha)
            - math.lgamma(n + 1 - m / alpha)
            - math.lgamma(n - k + 1)
            + math.lgamma(n + 1)
        )
        moments.append(math.pow(scale, m) * math.exp(log_ratio))
    return moments


def branch_tail(family: str, spec: Any) -> Callable[[float], float]:
    """
    P(X > t) for one fork-join branch, given its (family, spec):
    - "pareto": spec is a ParetoParams -- exact, no fitting.
    - "gamma"/"h2"/"erlang": spec is a list of raw moments -- fitted via that
      family's get_params (same moment-matching MaxDistribution already
      uses for the i.i.d. case).
    """
    if family == "pareto":
        # ParetoDistribution.get_tail assumes t >= K (the distribution's minimum
        # support value); below K, P(X>t) = 1 exactly, not the unclamped formula.
        return lambda t: 1.0 if t < spec.K else ParetoDistribution.get_tail(spec, t)
    if family == "gamma":
        params = GammaDistribution.get_params(spec)
        return lambda t: 1.0 - GammaDistribution.get_cdf(params, t)
    if family == "h2":
        params = H2Distribution.get_params(spec)
        return lambda t: 1.0 - H2Distribution.get_cdf(params, t)
    if family == "erlang":
        params = ErlangDistribution.get_params(spec)
        return lambda t: 1.0 - ErlangDistribution.get_cdf(params, t)
    raise ValueError(f"unknown family {family!r}; expected 'pareto', 'gamma', 'h2' or 'erlang'")


def heterogeneous_max_moments(branches: list[BranchSpec], num: int) -> list[float]:
    """
    Raw moments E[max(X_1,...,X_n)^k], k=1..num, of the maximum of `n`
    INDEPENDENT but not necessarily identically distributed random variables.

    Unlike ``pareto_max_moments`` (exact closed form, i.i.d. only), there is
    no closed form for the heterogeneous case in general -- but the moments
    are still numerically exact given each branch's assumed family, via the
    standard identity for a non-negative random variable Y = max(X_i):

        P(Y > t) = 1 - prod_i (1 - P(X_i > t))
        E[Y^k]   = k * integral_0^inf t^(k-1) * P(Y > t) dt

    evaluated with scipy.integrate.quad (the same numerical-integration
    standard used elsewhere in the library, e.g.
    most_queue.theory.srpt.utils.predictor). Reduces to ``pareto_max_moments``
    when all branches are the same Pareto(alpha, K) -- see
    docs/roadmaps/fork_join_dag_heterogeneous_roadmap.md sec. 1.

    :param branches: list of (family, spec) -- see ``branch_tail``.
    :param num: number of raw moments to compute.
    """
    if not branches:
        raise ValueError("branches must be non-empty")
    tails = [branch_tail(family, spec) for family, spec in branches]
    # Pareto branches have a hard kink in their tail at t=K (P(X>t)=1 below it);
    # quad's adaptive integration over the full [0, inf) struggles with a kink
    # this close to the singular endpoint, so split there explicitly.
    kinks = sorted({spec.K for family, spec in branches if family == "pareto"})

    def surv_max(t: float) -> float:
        p = 1.0
        for tail in tails:
            p *= 1.0 - tail(t)
        return 1.0 - p

    moments = []
    for k in range(1, num + 1):
        integrand = lambda t, k=k: k * t ** (k - 1) * surv_max(t)  # noqa: E731
        value = 0.0
        bounds = [0.0, *kinks, math.inf]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            # the last (semi-infinite) piece needs a much larger subdivision
            # budget for moments near a heavy-tailed branch's existence
            # boundary (k close to alpha), where the integrand decays slowly.
            part, _ = quad(integrand, lo, hi, limit=200 if hi != math.inf else 2000)
            value += part
        moments.append(value)
    return moments


def _poisson_binomial_at_least(probs: list[float], m: int) -> float:
    """
    P(at least m of n independent Bernoulli(p_i) events occur), via Rushdi's
    O(n^2) DP for the Poisson-Binomial distribution (Rushdi A.M., Recursive
    algorithm for reliability evaluation of a k-out-of-n:G system, IEEE
    Trans. Reliability, 1985, doi:10.1109/tr.1985.5221975 -- the same DP used
    for k-out-of-n:G system reliability with heterogeneous components).

        Q_0(0) = 1, Q_0(j) = 0 for j > 0
        Q_i(j) = Q_{i-1}(j)*(1-p_i) + Q_{i-1}(j-1)*p_i

    P(at least m) = sum_{j=m}^{n} Q_n(j).
    """
    n = len(probs)
    if m <= 0:
        return 1.0
    if m > n:
        return 0.0
    q = [1.0] + [0.0] * n  # q[j] = Q_i(j), updated in place as i grows
    for p in probs:
        for j in range(n, 0, -1):
            q[j] = q[j] * (1.0 - p) + q[j - 1] * p
        q[0] *= 1.0 - p
    return sum(q[m:])


def heterogeneous_kth_order_moments(branches: list[BranchSpec], k: int, num: int) -> list[float]:
    """
    Raw moments E[X_(k)^m], m=1..num, of the k-th order statistic (ascending:
    k=1 the minimum, k=n the maximum) of n INDEPENDENT but not necessarily
    identically distributed random variables.

    P(X_(k) <= x) = P(at least k of the n branches have X_i <= x) -- e.g.
    k=n (max) requires ALL n to have finished, matching P(max<=x)=prod(CDF_i(x));
    k=1 (min) requires at least 1 to have finished. Same DP as the
    k-out-of-n:G system reliability problem with heterogeneous component
    lifetimes (``_poisson_binomial_at_least``). Moments via the same
    tail-integral identity as ``heterogeneous_max_moments`` (k=n reduces to
    it exactly -- regression test).

    :param branches: list of (family, spec) -- see ``branch_tail``.
    :param k: order (1 <= k <= n, ascending: k=1 min, k=n max).
    :param num: number of raw moments to compute.
    """
    n = len(branches)
    if n == 0:
        raise ValueError("branches must be non-empty")
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")
    tails = [branch_tail(family, spec) for family, spec in branches]
    kinks = sorted({spec.K for family, spec in branches if family == "pareto"})

    def surv_kth(t: float) -> float:
        probs = [1.0 - tail(t) for tail in tails]  # P(X_i <= t)
        return 1.0 - _poisson_binomial_at_least(probs, k)

    moments = []
    for m in range(1, num + 1):
        integrand = lambda t, m=m: m * t ** (m - 1) * surv_kth(t)  # noqa: E731
        value = 0.0
        bounds = [0.0, *kinks, math.inf]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            part, _ = quad(integrand, lo, hi, limit=200 if hi != math.inf else 2000)
            value += part
        moments.append(value)
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
