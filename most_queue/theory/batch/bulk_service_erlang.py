"""
M/Erlang(k,rate)^[a,b]/1 bulk-service queue: general (not exponential)
batch-service time, fitted to an Erlang(k, rate) phase-type representation
-- closes the reserve flagged since EPIC-012 ("M/G^[b]/1 through an
embedded chain"). See docs/research/bulk-service-general-erlang-2026.md for
why the classical embedded-chain/PGF route (Neuts 1975, Chaudhry-Templeton)
was set aside in favor of this lower-risk, already-proven-elsewhere
technique: fit the general service distribution to a phase-type family
from its raw moments, and augment BulkServiceMM1Calc's exact CTMC (EPIC-012)
with a phase dimension, exactly like MaxDistribution/the SLA layer already
do for other "general distribution" extensions in this library.

State: (i, p, j) -- i = batch size in service (0 = idle), p = current
Erlang phase (0..k-1, meaningful only when i>=1), j = number waiting.
k=1 (Erlang collapses to Exponential) reduces EXACTLY to BulkServiceMM1Calc
-- verified to full float64 precision, the primary regression test.

A units/convention bug was caught before shipping: the phase-transition
rate was first set to k*rate (intending the *aggregate* rate across k
phases), which actually gives a total Erlang mean of 1/rate, not k/rate --
off by a factor of k^2 on the mean. Caught because it coincidentally still
passed the k=1 regression check (k*rate=rate when k=1) but diverged sharply
from an independent DES at k=3; fixed by using the per-phase rate directly
(rate, giving the standard Erlang(k,rate) mean k/rate).

Mean-only (E[N], E[V], E[W] via Little's law) was the original scope,
matching the level EPIC-012 shipped for the exponential case before
EPIC-032 added exact moments as a separate follow-up.

EPIC-042 adds batch-size-dependent rate: `rate` may be a callable
`rate(batch_size) -> per-phase rate`, porting BulkServiceMM1Calc's existing
callable-mu convention (LLM/GPU dynamic batching: a bigger batch takes
longer per phase, but serves more requests at once). `k` (phase count)
stays fixed across batch sizes -- only the per-phase rate varies -- to
avoid the state-space complexity of a variable phase count per batch size
(a materially harder, separate reserve item, same class of difficulty as
EPIC-041's per-server phase counts). `mean_batch_service` for `get_w()`
generalizes from the scalar `k/rate` to a busy-time-weighted average of
`k/rate(i)` over the batch-size distribution actually observed in the
solved stationary distribution.

EPIC-043 adds EXACT raw moments of W (not just the mean), for a=1, porting
EPIC-032's PASTA/hypoexponential-decomposition technique from the plain
exponential BulkServiceMM1Calc to this phase-augmented case: an arriving
customer sees the stationary (i, p, j); if busy (i>=1), their wait is the
REMAINING service of the current batch -- Erlang(k-p, rate(i)), the k-p
phases not yet completed -- plus j//b FULL batches ahead (each always
exactly size b, since batches are served FCFS taking b at a time until
fewer than b remain -- independent of which phase-dependent rate is used),
each Erlang(k, rate(b)), convolved via conv_moments. Validated against an
independent DES tracking each customer's own wait (not sojourn -- an
earlier validation attempt accidentally compared against sojourn time
instead of wait time, which looked like a bug until the DES was fixed to
record batch-START time rather than batch-COMPLETION time).

EPIC-067 generalizes to any ``1 <= a <= b``, porting
``BulkServiceMM1Calc``'s idle-refill race-aware decomposition: when the
remainder after the full batches ahead (plus the tagged customer) is below
the threshold ``a``, the wait is a RACE between new Poisson(lambda) arrivals
and the remaining ``(k - p) + full_ahead * k`` service phases (NOT a naive
"service phases, then a separate Erlang(need, lambda) refill" split -- see
``most_queue.theory.batch._idle_refill`` and
docs/epics/EPIC-067-bulk-service-idle-refill.md for why that is wrong and
what the correct 2-D construction is).
"""

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.batch._idle_refill import abandonment_chain, race_subgen
from most_queue.theory.utils.conv import conv_moments


class BulkServiceErlangCalc(BaseQueue):
    """
    Exact M/Erlang(k,rate)^[a,b]/1 bulk-service queue via a truncated,
    phase-augmented CTMC.

    :param a: minimum batch size to start a service.
    :param b: maximum batch size.
    :param k: number of Erlang phases (k=1 is Exponential -- exact
        regression to BulkServiceMM1Calc).
    :param queue_truncation: cap on the number waiting (state-space bound).
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, a: int, b: int, k: int, queue_truncation: int = 300, gamma: float = 0.0
    ):
        super().__init__(n=1)
        if not 1 <= a <= b:
            raise ValueError("require 1 <= a <= b")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.a = a
        self.b = b
        self.k = k
        self.N = queue_truncation
        self.gamma = gamma
        self.l = None
        self.rate_fn = None
        self._pi: np.ndarray | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, rate):  # pylint: disable=arguments-differ
        """
        :param rate: per-phase Erlang rate; total mean batch-service time = k / rate.
            A scalar (batch-size-independent), or a callable rate(batch_size) -> rate
            (e.g. LLM batching where a bigger batch is slower per phase).
        """
        if callable(rate):
            self.rate_fn = rate
        else:
            if rate <= 0:
                raise ValueError(f"rate must be positive, got {rate}")
            self.rate_fn = lambda _i, _rate=float(rate): _rate
        self.is_servers_set = True

    def set_servers_from_moments(self, moments: list[float]):
        """Fit (k, rate) from raw moments (mean, second moment, ...) via ErlangDistribution.get_params."""
        params: ErlangParams = ErlangDistribution.get_params(moments)
        self.k = params.r
        self.set_servers(params.mu)

    def _idle_index(self, j: int) -> int:
        return j

    def _busy_index(self, i: int, p: int, j: int) -> int:
        return (self.N + 1) + ((i - 1) * self.k + p) * (self.N + 1) + j

    def _solve_pi(self) -> np.ndarray:
        """Build and solve the CTMC; cache and return the stationary distribution pi."""
        if self._pi is not None:
            return self._pi
        self._check_if_servers_and_sources_set()

        a, b, k, N, lam, rate_fn = self.a, self.b, self.k, self.N, self.l, self.rate_fn
        gamma = self.gamma
        n_states = (N + 1) + b * k * (N + 1)
        rows, cols, vals = [], [], []

        def add(src, dst, r):
            rows.append(src)
            cols.append(dst)
            vals.append(r)

        for j in range(N + 1):
            s = self._idle_index(j)
            if j + 1 < a:
                add(s, self._idle_index(j + 1), lam)
            else:  # threshold reached -> batch of size a starts, phase 0
                add(s, self._busy_index(a, 0, 0), lam)
            if j >= 1 and gamma > 0:
                add(s, self._idle_index(j - 1), j * gamma)

        for i in range(1, b + 1):
            rate = rate_fn(i)
            for p in range(k):
                for j in range(N + 1):
                    s = self._busy_index(i, p, j)
                    if j < N:
                        add(s, self._busy_index(i, p, j + 1), lam)
                    if p < k - 1:
                        add(s, self._busy_index(i, p + 1, j), rate)
                    else:  # last phase -> batch completes
                        if j >= a:
                            take = min(b, j)
                            add(s, self._busy_index(take, 0, j - take), rate)
                        else:
                            add(s, self._idle_index(j), rate)
                    if j >= 1 and gamma > 0:
                        add(s, self._busy_index(i, p, j - 1), j * gamma)

        q = sp.coo_matrix((vals, (rows, cols)), shape=(n_states, n_states)).tocsr()
        out = np.asarray(q.sum(axis=1)).ravel()
        q = q - sp.diags(out)

        mat = q.transpose().tolil()
        mat[0, :] = 1.0
        rhs = np.zeros(n_states)
        rhs[0] = 1.0
        pi = spla.spsolve(mat.tocsr(), rhs)
        pi = np.maximum(np.real(pi), 0.0)
        pi = pi / pi.sum()
        self._pi = pi
        return pi

    def _abandon_chain_for_state(self, rate_first: float, n_first: int, j: int, start_idle: bool = False):
        """Build the EPIC-068 abandonment chain for a given observed state."""
        b, k, lam, gamma, rate_fn = self.b, self.k, self.l, self.gamma, self.rate_fn
        n_first = 0 if start_idle else n_first
        k_ahead = 0 if start_idle else k
        return abandonment_chain(
            rate_first, n_first, rate_fn(b), k_ahead, j, self.a, b, lam, gamma, start_idle=start_idle
        )

    def get_abandonment_prob(self) -> float:
        """
        EPIC-068: exact probability that a PASTA-arriving (tagged) customer
        abandons (Markovian patience, rate ``gamma``) before their own batch
        starts service. 0.0 when ``gamma == 0``. See
        ``BulkServiceMM1Calc.get_abandonment_prob`` for the full derivation
        (ahead-of-tagged customers can also abandon -- a genuine race, not a
        simple competing-exponential add-on to the EPIC-067 chain).
        """
        if self.gamma <= 0:
            return 0.0
        pi = self._solve_pi()
        a, b, k, rate_fn = self.a, self.b, self.k, self.rate_fn
        cache: dict[tuple, float] = {}
        p_abandon = 0.0
        for j in range(self.N + 1):
            prob = pi[self._idle_index(j)]
            if prob <= 0 or j == a - 1:
                continue
            key = ("idle", j)
            cached = cache.get(key)
            if cached is None:
                subgen, m, start, _ = self._abandon_chain_for_state(0.0, 0, j, start_idle=True)
                alpha = np.zeros(m)
                alpha[start] = 1.0
                x = spla.spsolve((-subgen).tocsc(), np.ones(m))
                cached = self.gamma * float(alpha @ x)
                cache[key] = cached
            p_abandon += prob * cached

        for i in range(1, b + 1):
            rate_i = rate_fn(i)
            for p in range(k):
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, p, j)]
                    if prob <= 0:
                        continue
                    key = ("busy", rate_i, k - p, j)
                    cached = cache.get(key)
                    if cached is None:
                        subgen, m, start, _ = self._abandon_chain_for_state(rate_i, k - p, j)
                        alpha = np.zeros(m)
                        alpha[start] = 1.0
                        x = spla.spsolve((-subgen).tocsc(), np.ones(m))
                        cached = self.gamma * float(alpha @ x)
                        cache[key] = cached
                    p_abandon += prob * cached
        return p_abandon

    def get_w(self, num: int = 4) -> list[float]:
        """
        Exact raw moments of W (waiting time) CONDITIONAL on being served, any
        ``1 <= a <= b``. At ``gamma == 0`` this is the unconditional W (see
        module docstring for the idle-refill decomposition); at ``gamma > 0``
        (EPIC-068) see ``BulkServiceMM1Calc.get_w`` for what "conditional"
        means here and why.
        """
        if self.gamma > 0:
            return self._get_w_with_abandonment(num)
        pi = self._solve_pi()
        a, b, k, lam, rate_fn = self.a, self.b, self.k, self.l, self.rate_fn

        def exp_moments(rate: float) -> list[float]:
            return [math.factorial(n) / rate**n for n in range(1, num + 1)]

        def erlang_moments(n_terms: int, rate: float) -> list[float]:
            if n_terms <= 0:
                return [0.0] * num
            acc = exp_moments(rate)
            for _ in range(n_terms - 1):
                acc = list(conv_moments(acc, exp_moments(rate), num))
            return acc

        moment_cache: dict[tuple, list[float]] = {}

        def race_moments(service_rates: np.ndarray, need: int) -> list[float]:
            key = (tuple(service_rates), need)
            cached = moment_cache.get(key)
            if cached is not None:
                return cached
            subgen, m, start = race_subgen(service_rates, lam, need)
            alpha = np.zeros(m)
            alpha[start] = 1.0
            neg_a = (-subgen).tocsc()
            x = np.ones(m)
            moments = []
            for n in range(1, num + 1):
                x = spla.spsolve(neg_a, x)
                moments.append(math.factorial(n) * float(alpha @ x))
            moment_cache[key] = moments
            return moments

        w_moments = np.zeros(num)
        for j in range(self.N + 1):
            prob = pi[self._idle_index(j)]
            if prob > 0:
                w_moments += prob * np.array(erlang_moments(a - j - 1, lam))

        for i in range(1, b + 1):
            rate_i = rate_fn(i)
            for p in range(k):
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, p, j)]
                    if prob <= 0:
                        continue
                    full_ahead = j // b
                    remainder = j - full_ahead * b
                    need = a - remainder - 1
                    if need > 0:
                        service_rates = np.full(k - p + full_ahead * k, rate_fn(b))
                        service_rates[: k - p] = rate_i
                        wm = race_moments(service_rates, need)
                    else:
                        wm = erlang_moments(k - p, rate_i)  # remaining phases of current batch
                        if full_ahead > 0:  # each full batch ahead is always exactly size b
                            wm = list(conv_moments(wm, erlang_moments(full_ahead * k, rate_fn(b)), num))
                    w_moments += prob * np.array(wm)
        return list(w_moments)

    def _get_w_with_abandonment(self, num: int = 4) -> list[float]:
        """EPIC-068: exact raw moments of W conditional on being served, gamma > 0."""
        pi = self._solve_pi()
        a, b, k, rate_fn = self.a, self.b, self.k, self.rate_fn
        cache: dict[tuple, tuple] = {}
        numerator = np.zeros(num)
        p_served_total = 0.0

        def accumulate(prob, key, builder):
            nonlocal p_served_total, numerator
            cached = cache.get(key)
            if cached is None:
                subgen, m, start, serve_rate = builder()
                alpha = np.zeros(m)
                alpha[start] = 1.0
                neg_a = (-subgen).tocsc()
                # E[W^n; served] = n! * alpha @ (-A)^-(n+1) @ serve_rate -- see
                # BulkServiceMM1Calc._get_w_with_abandonment for why the extra
                # (-A)^-1 application (vs. the gamma==0 "n applications to 1" moment
                # formula) is required; omitting it silently collapsed every moment
                # to p_served_state, caught via a full-system DES cross-check.
                x = spla.spsolve(neg_a, serve_rate)
                moments = []
                for n in range(1, num + 1):
                    x = spla.spsolve(neg_a, x)
                    moments.append(math.factorial(n) * float(alpha @ x))
                p_served_state = 1.0 - self.gamma * float(alpha @ spla.spsolve(neg_a, np.ones(m)))
                cached = (moments, p_served_state)
                cache[key] = cached
            moments, p_served_state = cached
            numerator += prob * np.array(moments)
            p_served_total += prob * p_served_state

        for j in range(self.N + 1):
            prob = pi[self._idle_index(j)]
            if prob <= 0:
                continue
            if j == a - 1:
                p_served_total += prob
                continue
            accumulate(prob, ("idle", j), lambda j=j: self._abandon_chain_for_state(0.0, 0, j, start_idle=True))

        for i in range(1, b + 1):
            rate_i = rate_fn(i)
            for p in range(k):
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, p, j)]
                    if prob <= 0:
                        continue
                    accumulate(
                        prob,
                        ("busy", rate_i, k - p, j),
                        lambda rate_i=rate_i, p=p, j=j: self._abandon_chain_for_state(rate_i, k - p, j),
                    )
        if p_served_total <= 0:
            return [0.0] * num
        return list(numerator / p_served_total)

    def get_tail(self, t: float) -> float:
        """
        Exact P(W > t), any ``1 <= a <= b``.

        Same decomposition as ``get_w`` (remaining ``k - p`` phases of the
        current batch at ``rate_fn(i)``, then ``full_ahead * k`` phases at
        ``rate_fn(b)``, then an idle-refill race if the remainder is still
        short of the threshold ``a`` -- see module/``_idle_refill``
        docstrings), evaluated via sparse matrix-exponential-action instead
        of moment convolution (see ``BulkServiceMM1Calc.get_tail`` for why
        sparse, not dense ``expm``). k=1 reduces exactly to
        ``BulkServiceMM1Calc.get_tail``.
        """
        if t < 0:
            raise ValueError("t must be nonnegative")
        if self.gamma > 0:
            return self._get_tail_with_abandonment(t)
        pi = self._solve_pi()
        a, b, k, lam, rate_fn = self.a, self.b, self.k, self.l, self.rate_fn
        cache: dict[tuple, float] = {}

        tail = 0.0
        for j in range(self.N + 1):
            prob = pi[self._idle_index(j)]
            need = a - j - 1
            if prob > 0 and need > 0:
                key = ("idle", need)
                cached = cache.get(key)
                if cached is None:
                    rates = np.full(need, lam)
                    subgen = sp.diags([-rates, rates[:-1]], [0, 1], format="csc")
                    cached = float(spla.expm_multiply(subgen * t, np.ones(need))[0])
                    cache[key] = cached
                tail += prob * cached

        for i in range(1, b + 1):
            rate_i = rate_fn(i)
            for p in range(k):
                remaining_current = k - p
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, p, j)]
                    if prob <= 0:
                        continue
                    full_ahead = j // b
                    remainder = j - full_ahead * b
                    need = a - remainder - 1
                    key = (rate_i, remaining_current, full_ahead, max(need, 0))
                    cached = cache.get(key)
                    if cached is None:
                        ahead_phases = full_ahead * k
                        m = remaining_current + ahead_phases
                        rates = np.full(m, rate_fn(b))
                        rates[:remaining_current] = rate_i
                        if need > 0:
                            subgen, m, start = race_subgen(rates, lam, need)
                            cached = float(spla.expm_multiply(subgen * t, np.ones(m))[start])
                        else:
                            subgen = sp.diags([-rates, rates[:-1]], [0, 1], format="csc")
                            cached = float(spla.expm_multiply(subgen * t, np.ones(m))[0])
                        cache[key] = cached
                    tail += prob * cached
        return tail

    def _get_tail_with_abandonment(self, t: float) -> float:
        """EPIC-068: exact P(W > t AND served) / P(served), gamma > 0."""
        pi = self._solve_pi()
        a, b, k, rate_fn = self.a, self.b, self.k, self.rate_fn
        cache: dict[tuple, tuple] = {}
        numerator = 0.0
        p_served_total = 0.0

        def accumulate(prob, key, builder):
            nonlocal p_served_total, numerator
            cached = cache.get(key)
            if cached is None:
                subgen, m, start, serve_rate = builder()
                alpha = np.zeros(m)
                alpha[start] = 1.0
                neg_a = (-subgen).tocsc()
                v = spla.spsolve(neg_a, serve_rate)
                p_served_state = 1.0 - self.gamma * float(alpha @ spla.spsolve(neg_a, np.ones(m)))
                cached = (subgen, v, start, p_served_state)
                cache[key] = cached
            subgen, v, start, p_served_state = cached
            numerator += prob * float(spla.expm_multiply(subgen * t, v)[start])
            p_served_total += prob * p_served_state

        for j in range(self.N + 1):
            prob = pi[self._idle_index(j)]
            if prob <= 0:
                continue
            if j == a - 1:
                p_served_total += prob
                continue
            accumulate(prob, ("idle", j), lambda j=j: self._abandon_chain_for_state(0.0, 0, j, start_idle=True))

        for i in range(1, b + 1):
            rate_i = rate_fn(i)
            for p in range(k):
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, p, j)]
                    if prob <= 0:
                        continue
                    accumulate(
                        prob,
                        ("busy", rate_i, k - p, j),
                        lambda rate_i=rate_i, p=p, j=j: self._abandon_chain_for_state(rate_i, k - p, j),
                    )
        if p_served_total <= 0:
            return 0.0
        return numerator / p_served_total

    def get_cdf(self, t: float) -> float:
        """Exact P(W <= t) -- see ``get_tail`` for scope (a=1 only)."""
        return 1.0 - self.get_tail(t)

    def run(self) -> QueueResults:
        """Solve the CTMC; return waiting/sojourn moments (exact, any 1 <= a <= b)."""
        start = self._measure_time()
        pi = self._solve_pi()
        b, k, N, lam = self.b, self.k, self.N, self.l

        e_n = 0.0
        for j in range(N + 1):
            e_n += pi[self._idle_index(j)] * j
        for i in range(1, b + 1):
            for p in range(k):
                for j in range(N + 1):
                    prob = pi[self._busy_index(i, p, j)]
                    e_n += prob * (i + j)

        e_t = e_n / lam
        w = self.get_w()

        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=list(w),
            utilization=1.0 - float(sum(pi[self._idle_index(j)] for j in range(N + 1))),
        )
        self._set_duration(res, start)
        return res
