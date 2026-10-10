"""
M/M/1 whose service rate switches at a threshold on the NUMBER IN SYSTEM:
``mu_low`` while at most ``K`` customers are present, ``mu_high`` above that.

Implementation of
    Morrison J.A., "Sojourn and waiting times in a single-server system with
    state-dependent mean service rate", Queueing Systems 4:213-235, 1989,
    doi:10.1007/BF02100267.

THE MODEL AND THE EXACT RESULT ARE HIS. Morrison formulates the general
state-dependent birth-death case and then solves the two-level threshold one,
obtaining a generating function for the Laplace transforms of the sojourn- and
waiting-time densities and explicit second moments.

A note on sources, so the provenance is not overstated. Morrison's paper is
paywalled and was NOT read. The derivation below was done from scratch and then
checked against the open modern treatment of the same model,
    Adan I., D'Auria B., "Sojourn time in a single server queue with threshold
    service rate control", SIAM J. Appl. Math. 76(1):197-216, 2016,
    arXiv:1509.04111 (their "continuous inspection" case, gamma = infinity),
whose equations (4) and (6) it reproduces -- including the boundary condition,
which is the one step that is easy to get wrong. Their random-inspection
extension (gamma < infinity, the rate changes only at Poisson inspection
epochs) is a different model and is NOT implemented here.

Why the sojourn time is hard here while the queue length is trivial. The number
in system is an ordinary birth-death chain, so its stationary distribution is a
one-liner. The sojourn time is not, because the rate depends on the TOTAL in
system -- including customers who arrive BEHIND the tagged one. A customer's own
service can therefore speed up or slow down because of arrivals that have no
other effect on it, which is exactly why Little's distributional law does not
apply and why the problem needed a paper.

The construction used here. Track the tagged customer as ``(r, N)``: ``r``
customers still ahead of it, ``N`` in system in total. Arrivals take
``N -> N+1``; a service completion takes ``N -> N-1`` and ``r -> r-1``, and is
the tagged customer's own departure when ``r = 0``. Writing the standard
moment recursion for an absorbing chain, ``(-A) m_k = k m_{k-1}``, gives

    (lam + mu(N)) v_k(r,N) = k v_{k-1}(r,N) + lam v_k(r,N+1) + mu(N) v_k(r-1,N-1)

What makes this FINITE rather than an infinite system is the following
observation. From ``(r, N)`` the tagged customer leaves after exactly ``r+1``
departures, so before it leaves the system never drops below ``N - r``. If
``N - r > K`` the rate is ``mu_high`` for the whole sojourn no matter how many
customers arrive, and the sojourn is exactly ``Erlang(r+1, mu_high)``. So each
level ``r`` needs only the ``K`` states ``N = r+1 .. K+r``, closed by that
boundary -- there is no truncation error in ``N`` at all. (In the notation of
Adan & D'Auria the same condition reads ``m >= K``, ``m`` being the number
behind.) The waiting time is the same recursion stopped one departure earlier.

Stability is governed by ``mu_high`` alone: ``lam < mu_high``. ``mu_low`` may be
below ``lam`` -- the queue simply grows until it crosses the threshold and is
then drained at the higher rate. The same structural remark applies to
``most_queue.theory.fifo.gi_m2_state_dependent``.
"""

import math

import numpy as np

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams

_MIN_LEVELS = 64  # never truncate the arrival distribution below this
_MAX_LEVELS = 1 << 16  # hard cap, so a near-unstable system fails loudly rather than hanging


def _erlang_moments(shape: int, rate: float, num: int) -> np.ndarray:
    """Raw moments of ``Erlang(shape, rate)``; ``shape = 0`` is the point mass at 0."""
    out = np.zeros(num)
    if shape == 0:
        return out
    for k in range(1, num + 1):
        out[k - 1] = math.factorial(shape + k - 1) / math.factorial(shape - 1) / rate**k
    return out


class MM1ThresholdRateCalc(BaseQueue):
    """
    M/M/1 with a two-level, threshold-controlled service rate (Morrison, 1989).

    :param threshold: ``K``. The server works at ``mu_low`` while the number in
        system is at most ``K`` and at ``mu_high`` above it. ``K = 0`` means the
        high rate always applies, so the system is an ordinary M/M/1.
    :param calc_params: standard calculation parameters; ``tolerance`` sets how
        far the arrival-state distribution is summed.
    """

    def __init__(self, threshold: int, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        self.threshold = int(threshold)
        self.l: float | None = None
        self.mu_low: float | None = None
        self.mu_high: float | None = None
        self._sol: dict | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = float(l)
        self._sol = None
        self.is_sources_set = True

    def set_servers(self, mu_low: float, mu_high: float):  # pylint: disable=arguments-differ
        """
        :param mu_low: service rate while the number in system is at most ``K``.
        :param mu_high: service rate above the threshold. ``mu_high > lam`` is
            the stability condition; ``mu_low`` is unconstrained and may be
            smaller than ``lam``.
        """
        if mu_low <= 0 or mu_high <= 0:
            raise ValueError(f"service rates must be positive, got mu_low={mu_low}, mu_high={mu_high}")
        self.mu_low = float(mu_low)
        self.mu_high = float(mu_high)
        self._sol = None
        self.is_servers_set = True

    # --------------------------------------------------------------- internals
    def _rate(self, n: int) -> float:
        return self.mu_low if n <= self.threshold else self.mu_high

    def _utilization(self) -> float:
        self._check_if_servers_and_sources_set()
        rho = self.l / self.mu_high
        if rho >= 1:
            raise ValueError(
                f"System is unstable: lam/mu_high = {rho} must be < 1. Stability depends on the "
                "above-threshold rate only; mu_low may be below lam."
            )
        return rho

    def _num_levels(self, num: int = 4) -> int:
        """
        How far to sum over the state an arrival finds.

        The probabilities decay like ``rho^n``, but the ``k``-th moment weights
        them by roughly ``(n/mu_high)^k``, so the number of terms has to grow
        with the moment order -- truncating on the probabilities alone leaves a
        visible error in the fourth moment.
        """
        rho = self._utilization()
        scale = max(1.0 / self.mu_high, 1.0)
        levels = max(self.threshold + 1, _MIN_LEVELS)
        while levels < _MAX_LEVELS:
            tail = rho**levels * ((levels + 1) * scale) ** num / (1.0 - rho)
            if tail < self.calc_params.tolerance * 1e-3:
                break
            levels *= 2
        return min(levels, _MAX_LEVELS)

    def _conditional_moments(self, num: int) -> tuple[np.ndarray, np.ndarray]:
        """
        ``(V, W)`` with ``V[n, k]`` the ``(k+1)``-th raw moment of the sojourn
        time of a customer that finds ``n`` in system, and ``W`` likewise for the
        waiting time. Exact: the recursion closes on the Erlang boundary.
        """
        lam, k_thr = self.l, self.threshold
        levels = self._num_levels(num)
        width = k_thr + 1  # K recursed states plus one boundary slot

        v_out = np.zeros((levels + 1, num))
        w_out = np.zeros((levels + 1, num))
        v_prev = w_prev = None

        for r in range(levels + 1):
            rates = np.array([self._rate(r + 1 + idx) for idx in range(width)])

            v_cur = np.zeros((num + 1, width))
            v_cur[0, :] = 1.0
            v_cur[1:, k_thr] = _erlang_moments(r + 1, self.mu_high, num)
            for k in range(1, num + 1):
                for idx in range(k_thr - 1, -1, -1):
                    rhs = k * v_cur[k - 1, idx] + lam * v_cur[k, idx + 1]
                    if r >= 1:
                        rhs += rates[idx] * v_prev[k, idx]
                    v_cur[k, idx] = rhs / (lam + rates[idx])
            v_out[r] = v_cur[1:, 0]

            w_cur = np.zeros((num + 1, width))
            w_cur[0, :] = 1.0
            if r >= 1:
                w_cur[1:, k_thr - 1 if k_thr >= 1 else 0] = _erlang_moments(r, self.mu_high, num)
                for k in range(1, num + 1):
                    for idx in range(k_thr - 2, -1, -1):
                        rhs = k * w_cur[k - 1, idx] + lam * w_cur[k, idx + 1] + rates[idx] * w_prev[k, idx]
                        w_cur[k, idx] = rhs / (lam + rates[idx])
            w_out[r] = w_cur[1:, 0]

            v_prev, w_prev = v_cur, w_cur
        return v_out, w_out

    def _solve(self, num: int = 4) -> dict:
        if self._sol is not None and self._sol["num"] >= num:
            return self._sol
        self._utilization()
        levels = self._num_levels(num)
        probs = self.get_p(levels + 1)
        v_cond, w_cond = self._conditional_moments(num)
        weights = np.array(probs[: levels + 1])
        self._sol = {
            "num": num,
            "v": [float(weights @ v_cond[:, k]) for k in range(num)],
            "w": [float(weights @ w_cond[:, k]) for k in range(num)],
            "v_cond": v_cond,
            "w_cond": w_cond,
        }
        return self._sol

    # ----------------------------------------------------------------- results
    def get_p(self, num: int | None = None) -> list[float]:
        """
        Stationary distribution of the number in system -- an ordinary
        birth-death chain, geometric with ratio ``lam/mu_low`` up to the
        threshold and ``lam/mu_high`` above it.
        """
        rho = self._utilization()
        count = num or self.calc_params.p_num
        unnormalised = [1.0]
        for n in range(1, max(count, self.threshold + 2)):
            unnormalised.append(unnormalised[-1] * self.l / self._rate(n))
        # The tail beyond the computed range is geometric with ratio rho.
        total = sum(unnormalised) + unnormalised[-1] * rho / (1.0 - rho)
        self.p = [x / total for x in unnormalised[:count]]
        return self.p

    def get_conditional_sojourn_moments(self, n: int, num: int = 4) -> list[float]:
        """Exact raw moments of the sojourn time of a customer that finds ``n`` in system."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        sol = self._solve(num)
        if n >= sol["v_cond"].shape[0]:
            raise ValueError(f"n={n} is beyond the computed range {sol['v_cond'].shape[0] - 1}")
        return [float(x) for x in sol["v_cond"][n, :num]]

    def get_conditional_waiting_moments(self, n: int, num: int = 4) -> list[float]:
        """Exact raw moments of the waiting time of a customer that finds ``n`` in system."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        sol = self._solve(num)
        if n >= sol["w_cond"].shape[0]:
            raise ValueError(f"n={n} is beyond the computed range {sol['w_cond'].shape[0] - 1}")
        return [float(x) for x in sol["w_cond"][n, :num]]

    def get_v(self, num: int = 4) -> list[float]:
        """Exact raw moments of the sojourn time."""
        self.v = self._solve(num)["v"][:num]
        return self.v

    def get_w(self, num: int = 4) -> list[float]:
        """Exact raw moments of the waiting time."""
        self.w = self._solve(num)["w"][:num]
        return self.w

    def get_wait_prob(self) -> float:
        """``P(W > 0)`` -- by PASTA, the probability an arrival finds a busy server."""
        return 1.0 - self.get_p(2)[0]

    def get_n_mean(self) -> float:
        """Mean number in system, summed in closed form over the geometric tail."""
        rho = self._utilization()
        probs = self.get_p(self.threshold + 2)
        head = sum(n * x for n, x in enumerate(probs[: self.threshold + 1]))
        # Above the threshold p_n = p_{K+1} rho^(n-K-1), so the tail sums exactly:
        # sum_{j>=0} (K+1+j) rho^j = (K+1)/(1-rho) + rho/(1-rho)^2.
        start = self.threshold + 1
        edge = probs[start] if len(probs) > start else 0.0
        tail = edge * (start / (1.0 - rho) + rho / (1.0 - rho) ** 2)
        return float(head + tail)

    def get_service_time_mean(self) -> float:
        """
        Mean time actually spent in service, ``E[V] - E[W]``.

        It lies between ``1/mu_low`` and ``1/mu_high`` and equals neither: a
        customer may be served partly below and partly above the threshold.
        """
        v = self.v if self.v is not None else self.get_v(1)
        w = self.w if self.w is not None else self.get_w(1)
        return v[0] - w[0]

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """Solve and report the standard metrics."""
        start = self._measure_time()
        with self._validate_state():
            utilization = self._utilization()
            p = self.get_p()
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
        result = QueueResults(v=v, w=w, p=p, utilization=utilization)
        self._set_duration(result, start)
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(K={self.threshold}, mu_low={self.mu_low}, mu_high={self.mu_high})"
