"""
GI/M/2 in which the service rate depends on how many servers are busy.

Implementation of
    Bhat U.N., "The queue GI/M/2 with service rate depending on the number of
    busy servers", Annals of the Institute of Statistical Mathematics 18,
    1966, pp. 211-221, doi:10.1007/BF02869531.

THE MODEL AND THE SOLUTION ARE HIS. Equation numbers below refer to that paper.
Bhat solves the TIME-DEPENDENT behaviour, in double transforms; what is
implemented here is the steady state, obtained as his section 5 prescribes --
``P_j = lim_{theta->0} theta * phi_{0j}(theta)`` applied to his (45)-(47). He
carries that limit out only for Poisson arrivals at three particular values of
sigma (his table (56)), which is exactly what this implementation is checked
against.

The mechanism. Two servers share one FCFS queue. While both are busy each works
at rate ``mu``; while only one is busy it works at rate ``mu_single``, which
need not equal ``mu``. Bhat's motivation is two repairmen who help each other
when one is free (``mu_single > mu``, acceleration), and the opposite case where
a lone server slows down. Two values of the ratio are degenerate and make good
regression targets:

- ``mu_single == mu`` is the ordinary GI/M/2;
- ``mu_single == 2*mu`` leaves the total capacity unchanged at every occupancy,
  so the system IS a GI/M/1 with rate ``2*mu``.

Arrivals are a general renewal process, so PASTA does NOT hold: what an arrival
sees is not what a random instant sees, and both are produced here
(``get_pi`` and ``get_p``).

One structural consequence worth naming, since it is not obvious and is not in
the paper. The root ``gamma`` of ``z = psi(2 mu (1-z))`` involves only the
both-busy rate, so it does not depend on ``mu_single`` at all. The waiting time
therefore comes out as an atom at zero plus ``Exp(2 mu (1 - gamma))`` -- the
same exponential as in an ordinary GI/M/2. **The state dependence changes how
OFTEN a customer waits, not how long it waits once it does.** All the
``mu_single`` dependence sits in the probability of the atom.
"""

import math

from scipy.optimize import brentq

from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.params import ErlangParams, GammaParams, H2Params
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


def _lst(params, s: float) -> float:
    """Laplace-Stieltjes transform of the fitted interarrival distribution."""
    if isinstance(params, GammaParams):
        return float(GammaDistribution.get_lst(params, s))
    if isinstance(params, H2Params):
        return float(params.p1 * params.mu1 / (params.mu1 + s) + (1 - params.p1) * params.mu2 / (params.mu2 + s))
    if isinstance(params, ErlangParams):
        return float((params.mu / (params.mu + s)) ** params.r)
    raise TypeError(f"no Laplace-Stieltjes transform available for {type(params).__name__}")


def _mean(params) -> float:
    """Mean of the fitted interarrival distribution."""
    if isinstance(params, GammaParams):
        return float(params.alpha / params.mu)
    if isinstance(params, H2Params):
        return float(params.p1 / params.mu1 + (1 - params.p1) / params.mu2)
    if isinstance(params, ErlangParams):
        return float(params.r / params.mu)
    raise TypeError(f"no mean available for {type(params).__name__}")


class GiM2StateDependentCalc(BaseQueue):
    """
    GI/M/2 with a service rate that depends on the number of busy servers
    (Bhat, 1966).

    :param calc_params: standard calculation parameters; ``p_num`` sets how many
        state probabilities are reported.
    """

    def __init__(self, calc_params: CalcParams | None = None):
        super().__init__(n=2, calc_params=calc_params)
        self.a: list[float] | None = None  # interarrival raw moments, if given
        self.arrival_params = None
        self.mu: float | None = None  # per-server rate while BOTH are busy
        self.mu_single: float | None = None  # rate while only ONE is busy
        self.pi: list[float] | None = None  # arrival-observed state probabilities
        self._sol: dict | None = None

    # ------------------------------------------------------------------ setup
    def set_sources(self, a: list[float]):  # pylint: disable=arguments-differ
        """
        :param a: raw moments of the interarrival time. A distribution is fitted
            to them following ``calc_params.approx_distr`` (gamma by default),
            the same convention ``GiMn`` uses -- the solution needs the
            interarrival transform, not just a few moments.
        """
        if not a or a[0] <= 0:
            raise ValueError("interarrival moments must be non-empty with a positive mean")
        if self.calc_params.approx_distr != "gamma":
            raise ValueError(
                f"only the gamma fit is supported here, got approx_distr="
                f"{self.calc_params.approx_distr!r}; use set_sources_params() to supply "
                "an H2 or Erlang interarrival distribution explicitly"
            )
        self.a = list(a)
        self.arrival_params = GammaDistribution.get_params(list(a))
        self._sol = None
        self.is_sources_set = True

    def set_sources_params(self, params):
        """
        Set the interarrival distribution explicitly, as
        :class:`GammaParams`, :class:`H2Params` or :class:`ErlangParams`.
        """
        self.arrival_params = params
        self.a = [_mean(params)]
        self._sol = None
        self.is_sources_set = True

    def set_servers(self, mu: float, mu_single: float | None = None):  # pylint: disable=arguments-differ
        """
        :param mu: service rate of each server while BOTH are busy.
        :param mu_single: rate of the lone busy server while only one is busy.
            Defaults to ``mu`` -- the ordinary GI/M/2. ``2*mu`` makes the system
            a GI/M/1 of rate ``2*mu``.
        """
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        mu_single = mu if mu_single is None else mu_single
        if mu_single <= 0:
            raise ValueError(f"mu_single must be positive, got {mu_single}")
        self.mu = float(mu)
        self.mu_single = float(mu_single)
        self._sol = None
        self.is_servers_set = True

    # --------------------------------------------------------------- internals
    def _utilization(self) -> float:
        """
        ``rho = arrival rate / (2 mu)``. Stability is set by the both-busy rate
        alone -- ``mu_single`` governs only the lightly loaded states, which a
        growing queue leaves behind.
        """
        self._check_if_servers_and_sources_set()
        rho = 1.0 / (self.a[0] * 2.0 * self.mu)
        if rho >= 1:
            raise ValueError(f"System is unstable: rho={rho} must be < 1 (arrival rate vs 2*mu)")
        return rho

    def _solve(self) -> dict:
        """The theta -> 0 limit of Bhat's (45)-(47), cached."""
        if self._sol is not None:
            return self._sol
        rho = self._utilization()
        mean_a = self.a[0]
        two_mu = 2.0 * self.mu
        sigma = self.mu_single / two_mu

        # (18) at theta = 0, omega = 1: the classical GI/M/2 root, independent of mu_single.
        gamma = brentq(lambda z: _lst(self.arrival_params, two_mu * (1.0 - z)) - z, 1e-14, 1.0 - 1e-14)
        psi2 = _lst(self.arrival_params, self.mu_single)

        denom = (1.0 - gamma) * (1.0 - sigma) - sigma * psi2
        if abs(denom) < 1e-14:
            raise ValueError(
                "the steady-state expression degenerates for these parameters "
                f"(sigma={sigma}, gamma={gamma}); perturb mu_single slightly"
            )
        pref = (1.0 - gamma) / (mean_a * denom)
        common = (1.0 - sigma) * (1.0 - psi2) / self.mu_single

        p1 = pref * (common - psi2 / two_mu)  # (46)
        p0 = 1.0 - pref * (common - sigma * psi2 / (two_mu * (1.0 - gamma)))  # (47)
        tail_base = (1.0 - gamma) * (1.0 - sigma - gamma) * psi2 / (two_mu * mean_a * denom)  # (45), j = 2

        self._sol = {
            "rho": rho,
            "gamma": gamma,
            "sigma": sigma,
            "p0": p0,
            "p1": p1,
            "tail_base": tail_base,
            "two_mu": two_mu,
            "mean_a": mean_a,
        }
        return self._sol

    # ----------------------------------------------------------------- results
    def get_p(self, num: int | None = None) -> list[float]:
        """
        Time-stationary distribution of the number in system, ``P_j``.

        Geometric with ratio ``gamma`` from ``j = 2`` on; the state dependence
        shows up only in ``P_0`` and ``P_1``.
        """
        s = self._solve()
        count = num or self.calc_params.p_num
        p = [s["p0"], s["p1"]]
        for j in range(2, count):
            p.append(s["tail_base"] * s["gamma"] ** (j - 2))
        self.p = p[:count]
        return self.p

    def get_pi(self, num: int | None = None) -> list[float]:
        """
        Distribution seen by an ARRIVING customer, ``pi_j``.

        Not the same as :meth:`get_p` -- the input is a general renewal process,
        so PASTA does not apply. Obtained by level crossing: in steady state the
        rate at which arrivals push the level from ``j`` to ``j+1`` equals the
        rate at which departures bring it back, so
        ``pi_j = alpha * (departure rate at j+1) * P_{j+1}``.
        """
        s = self._solve()
        count = num or self.calc_params.p_num
        p = self.get_p(count + 1)
        pi = [self.mu_single * p[1] * s["mean_a"]]
        pi.extend(s["two_mu"] * p[j + 1] * s["mean_a"] for j in range(1, count))
        self.pi = pi[:count]
        return self.pi

    def get_q(self) -> float:
        """Mean number of customers in system, summed in closed form."""
        s = self._solve()
        gamma = s["gamma"]
        # sum_{j>=2} j * base * gamma^(j-2) = base * (2 - gamma) / (1 - gamma)^2
        return float(s["p1"] + s["tail_base"] * (2.0 - gamma) / (1.0 - gamma) ** 2)

    def get_wait_prob(self) -> float:
        """Probability that an arriving customer has to wait at all."""
        s = self._solve()
        return float(s["two_mu"] * s["mean_a"] * s["tail_base"] * s["gamma"] / (1.0 - s["gamma"]))

    def get_w(self, num: int = 4) -> list[float]:
        """
        Exact raw moments of the waiting time.

        An arrival that finds ``j >= 2`` in system must wait for ``j-1``
        departures, and throughout that wait both servers are busy, so the wait
        is ``Erlang(j-1, 2mu)``. Since ``pi_j`` is geometric with ratio
        ``gamma``, the geometric mixture of Erlangs collapses to a single
        exponential: the wait is zero with probability ``pi_0 + pi_1`` and
        ``Exp(2 mu (1 - gamma))`` otherwise. Hence

            ``E[W^k] = P(wait) * k! / (2 mu (1 - gamma))^k``.
        """
        if num < 1:
            raise ValueError(f"num must be at least 1, got {num}")
        s = self._solve()
        rate = s["two_mu"] * (1.0 - s["gamma"])
        wait_prob = self.get_wait_prob()
        self.w = [wait_prob * math.factorial(k) / rate**k for k in range(1, num + 1)]
        return self.w

    def get_w_tail(self, t: float) -> float:
        """Exact ``P(W > t)``."""
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")
        s = self._solve()
        return float(self.get_wait_prob() * math.exp(-s["two_mu"] * (1.0 - s["gamma"]) * t))

    def get_w_cdf(self, t: float) -> float:
        """Exact ``P(W <= t)``."""
        return 1.0 - self.get_w_tail(t)

    def get_v(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean sojourn time, by Little's law: ``E[V] = E[Q] * alpha``.

        First moment only, deliberately. The customer's own service rate changes
        while it is being served -- it is ``mu_single`` whenever it is alone and
        ``mu`` otherwise -- so its service duration is not exponential and the
        higher sojourn moments do not follow from the waiting-time ones.
        """
        s = self._solve()
        self.v = [self.get_q() * s["mean_a"]]
        return self.v

    def get_service_time_mean(self) -> float:
        """
        Mean time a customer actually spends in service, ``E[V] - E[W]``.

        Differs from both ``1/mu`` and ``1/mu_single``: a customer may be served
        partly alone and partly alongside another, at different rates.
        """
        v = self.v if self.v is not None else self.get_v()
        w = self.w if self.w is not None else self.get_w(1)
        return v[0] - w[0]

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """Solve and report the standard metrics."""
        start = self._measure_time()
        with self._validate_state():
            utilization = self._utilization()
            p = self.get_p()
            pi = self.get_pi()
            w = self.get_w(num_of_moments)
            v = self.get_v()
        result = QueueResults(v=v, w=w, p=p, pi=pi, utilization=utilization)
        self._set_duration(result, start)
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu}, mu_single={self.mu_single})"
