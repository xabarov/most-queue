"""
M/M/1 with deadline-aware admission control (EPIC-034): FCFS service order
is kept (unlike EDF, theory.priority... no -- see most_queue.sim.edf,
EPIC-029, which reorders service and was found to have no exact
finite-state solution in general); instead, each arriving job draws its own
relative deadline D ~ Exp(theta) and is admitted iff D exceeds the current
workload (virtual waiting time) it would observe -- otherwise it is
rejected outright and never joins the queue at all (not reneging/
abandonment, which happens after joining).

Literature: Das S., Jenkins L., Sengupta D., Analysis of an M/M/1+G queue
operated under the FCFS policy with exact admission control, Queueing
Systems, 2013, doi:10.1007/s11134-013-9366-6 (paywalled; the derivation
below is independent). See docs/research/llm-serving-deadline-admission-control-2026.md
for why a much simpler-looking shortcut (a state-dependent birth-death
chain on the number in system n, with P(admit|n) = (mu/(mu+theta))^n) is
WRONG -- n is not a sufficient statistic, because conditioning on admission
size-biases the workload by the deadline's (non-constant) tail function,
and that bias is not erased by service memorylessness the way ordinary
elapsed-time conditioning is. The correct treatment needs the full
continuous workload process.

Exact solution via a level-crossing (Takacs-style) functional equation for
the stationary workload's Laplace-Stieltjes transform phi(s) = E[e^{-sU}]:

    phi(s) = pi0 + [lambda / (s + mu)] * phi(s + theta)

which -- because the deadline is Exp(theta), giving a *constant-ratio* tail
that collapses the weighted transform to a pure shift -- telescopes into a
series that converges for any lambda, mu, theta > 0 (the system is always
stable: large-workload arrivals get auto-rejected):

    phi(s) = pi0 * sum_{n=0}^inf  lambda^n / prod_{j=0}^{n-1} (s + j*theta + mu)
    pi0    = 1 / [phi(s)'s sum, evaluated at s=0, before the pi0 factor]

Raw moments are extracted via truncated power-series (Taylor) arithmetic
around s=0 -- exact term-by-term (no finite-difference error): the
coefficient of s^k in phi(s) is (-1)^k * E[U^k] / k!. Moments *given
admission* use the same trick applied to phi(s+theta) (whose s=0 Taylor
coefficients are phi's derivatives at theta), divided by phi(theta) (the
overall acceptance probability, since phi(theta) = E[e^{-theta U}] =
E[tail_D(U)] = P(D>U) by PASTA). Sojourn time of an admitted job,
V = U + S with S ~ Exp(mu) drawn fresh *after* admission (independent of U
and of the admission event), so V's admitted-conditional moments are the
exact convolution of U's admitted-conditional moments with Exp(mu)'s
(conv_moments). Validated against an independent, from-scratch,
continuous-workload DES (not an n-based one) -- see the research doc for
the numbers.
"""

import math

from most_queue.structs import AdmissionControlResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.conv import conv_moments


def _series_mul(a: list[float], b: list[float], num: int) -> list[float]:
    """Truncated Cauchy product of two power series (coefficients a[0..num], b[0..num])."""
    out = [0.0] * (num + 1)
    for i in range(num + 1):
        if a[i] == 0.0:
            continue
        for k in range(num + 1 - i):
            out[i + k] += a[i] * b[k]
    return out


def _series_reciprocal(a: list[float], num: int) -> list[float]:
    """Truncated power series of 1/a(s), a[0] != 0 (standard recursive division algorithm)."""
    h = [0.0] * (num + 1)
    h[0] = 1.0 / a[0]
    for k in range(1, num + 1):
        s = 0.0
        for i in range(1, k + 1):
            s += a[i] * h[k - i]
        h[k] = -s / a[0]
    return h


def _phi_taylor(offset: float, lam: float, mu: float, theta: float, num: int, n_terms: int) -> list[float]:
    """
    Taylor coefficients (around s=0, up to order `num`) of phi(s + offset),
    i.e. c[k] = phi^(k)(offset) / k!, via the series
    phi(s+offset) = sum_n lambda^n / prod_{j=0}^{n-1}(s + offset + j*theta + mu)
    (the pi0 normalization factor is NOT applied here -- see get_*).
    """
    total = [1.0] + [0.0] * num  # n=0 term: lambda^0 / (empty product) = 1
    # Accumulate 1/C_n(s) directly (multiplying by each new factor's own
    # reciprocal) rather than building up C_n(s) and inverting the whole
    # product at the end -- C_n(0) can overflow float64 for large n_terms
    # combined with a large theta (e.g. theta=1000, n_terms=150 gives
    # C_150(0) with ~730 digits) even though each individual series term
    # lambda^n/C_n(s) stays small and well-behaved.
    inv_c_n = [1.0] + [0.0] * num  # running product of per-factor reciprocals
    for n in range(1, n_terms):
        a_j = offset + (n - 1) * theta + mu
        linear = [a_j] + [0.0] * num
        if num >= 1:
            linear[1] = 1.0
        inv_c_n = _series_mul(inv_c_n, _series_reciprocal(linear, num), num)
        term = [lam**n * c for c in inv_c_n]
        total = [t + u for t, u in zip(total, term)]
    return total


def _taylor_to_moments(taylor: list[float]) -> list[float]:
    """E[X^k] = (-1)^k * k! * (Taylor coeff of s^k), k=1..len(taylor)-1."""
    return [((-1) ** k) * math.factorial(k) * taylor[k] for k in range(1, len(taylor))]


class MM1DeadlineAdmissionControlCalc(BaseQueue):
    """
    Exact M/M/1 with deadline-aware admission control, D ~ Exp(theta).

    :param num_terms: truncation of the convergent series for phi(s) --
        150 is generous for typical lambda/mu/theta; increase if `residual()`
        (change from doubling num_terms) is not yet negligible.
    """

    def __init__(self, num_terms: int = 150, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)
        if num_terms < 10:
            raise ValueError(f"num_terms must be >= 10, got {num_terms}")
        self.num_terms = num_terms
        self.l = None
        self.mu = None
        self.theta = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, mu: float):  # pylint: disable=arguments-differ
        """:param mu: service rate (Exp(mu))."""
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        self.mu = float(mu)
        self.is_servers_set = True

    def set_deadline(self, theta: float):
        """:param theta: relative-deadline rate, D ~ Exp(theta)."""
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.theta = float(theta)

    def _check_state(self):
        self._check_if_servers_and_sources_set()
        if self.theta is None:
            raise RuntimeError("set_deadline() must be called before run()")

    def _pi0(self) -> float:
        base = _phi_taylor(0.0, self.l, self.mu, self.theta, 0, self.num_terms)
        return 1.0 / base[0]

    def get_u_moments(self, num: int = 4) -> list[float]:
        """Exact raw moments of U (workload/virtual waiting time), unconditional."""
        self._check_state()
        pi0 = self._pi0()
        taylor = _phi_taylor(0.0, self.l, self.mu, self.theta, num, self.num_terms)
        taylor = [pi0 * c for c in taylor]
        return _taylor_to_moments(taylor)

    def get_loss_prob(self) -> float:
        """P(an arriving job is rejected outright) = 1 - phi(theta) (PASTA)."""
        self._check_state()
        pi0 = self._pi0()
        acceptance = pi0 * _phi_taylor(self.theta, self.l, self.mu, self.theta, 0, self.num_terms)[0]
        return 1.0 - acceptance

    def get_u_moments_admitted(self, num: int = 4) -> list[float]:
        """Exact raw moments of U, conditional on the arriving job being admitted."""
        self._check_state()
        pi0 = self._pi0()
        taylor = _phi_taylor(self.theta, self.l, self.mu, self.theta, num, self.num_terms)
        taylor = [pi0 * c for c in taylor]
        acceptance = taylor[0]
        weighted_moments = _taylor_to_moments(taylor)
        return [m / acceptance for m in weighted_moments]

    def run(self, num: int = 4) -> AdmissionControlResults:
        """
        Solve the model; v/w are the sojourn/wait moments of ADMITTED jobs
        (rejected jobs never join, so have no wait/sojourn to report).
        """
        start = self._measure_time()
        with self._validate_state():
            u_admitted = self.get_u_moments_admitted(num)
            s_moments = [math.factorial(k) / self.mu**k for k in range(1, num + 1)]
            v = list(conv_moments(u_admitted, s_moments, num))
            loss_prob = self.get_loss_prob()

        result = AdmissionControlResults(
            v=v,
            w=u_admitted,
            utilization=self.l * (1.0 - loss_prob) / self.mu,
            loss_prob=loss_prob,
        )
        self._set_duration(result, start)
        return result
