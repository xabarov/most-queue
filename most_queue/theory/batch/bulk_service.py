"""
M/M^[a,b]/1 bulk-service (batch-service) queue.

A single server serves customers in **batches**: it starts a service only when at
least `a` customers are waiting and then takes up to `b` of them; the whole batch
finishes together after an exponential batch-service time. This is the classic
general bulk-service rule (Neuts / Chaudhry-Templeton) and the base model for
request batching in LLM inference serving.

Solved as a finite CTMC on the state (i, j) where i is the size of the batch in
service (0 = idle) and j is the number waiting. The batch-service rate may depend
on the batch size (relevant for LLM batching, where a larger batch takes longer
but serves more requests at once).

EPIC-032 adds exact raw moments (not just the mean):

- ``get_n_moments`` -- moments of N (number in system), trivial from the
  already-solved stationary distribution pi, any a, b.
- ``get_w`` -- moments of W (waiting time), via PASTA: an arriving customer
  sees the stationary (i, j); if busy (i>=1) their wait is the remaining
  current batch (Exp(mu(i)), memoryless) plus j // b *full* batches of size
  b ahead (each Exp(mu(b)), independent) -- a hypoexponential distribution,
  moments via ``conv_moments``.

EPIC-067 generalizes this to any ``1 <= a <= b`` (closing the reserve flagged
above and in docs/research/bulk-service-waiting-moments-2026.md). If the
remainder after the full batches ahead is below the threshold ``a``, the
tagged customer's own batch cannot start the instant the batches ahead of it
clear.

**First (wrong) attempt, caught before shipping:** treat the shortfall as a
simple POST-completion wait of ``need = a - remainder - 1`` more Poisson
arrivals, appended as one more sequential Erlang(need, lambda) phase after
the batches-ahead phases -- i.e. assume NO new arrivals occur during the
remaining service of the batches ahead. Confirmed numerically wrong (17-30%
error, per-state trace against an independent simulator): arrivals occurring
WHILE the batches ahead are still in service also count toward the
threshold -- the true dynamics are a RACE between "a new arrival occurs"
(rate lambda) and "the batch-in-service completes" (rate mu), not a clean
sequential split.

**Correct construction:** a 2-D absorbing CTMC crossing "which service phase
are we waiting out" (0 = remaining current batch at mu(i), 1..full_ahead =
each full batch ahead at mu(b)) with "how many of the needed `need` extra
arrivals have occurred so far" (0..need, capped -- arrivals beyond `need`
don't change anything). From each (phase, count) cell: rate lambda advances
the count (if not yet capped); rate mu(phase) advances the phase. Reaching
the LAST phase with count==need absorbs directly (enough had already
accumulated once service freed up -- by memorylessness, no further wait).
Reaching the last phase with count<need falls through to a pure Erlang(need
- count, lambda) refill tail. Both tail (``expm_multiply`` on the 2-D
subgenerator) and moments (general phase-type formula
``n! * alpha @ (-A)^-n @ 1``, used only when this 2-D structure is actually
needed -- the simple sequential chain, via ``conv_moments``, remains exact
and cheaper whenever ``need <= 0``) use the SAME explicit subgenerator, built
by the shared ``most_queue.theory.batch._idle_refill.race_subgen`` (also
reused by ``BulkServiceErlangCalc``). See
docs/epics/EPIC-067-bulk-service-idle-refill.md for the full derivation and
validation (independent per-state trace against ``BulkServiceSim``/a
from-scratch sampler).
"""

import math
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.batch._idle_refill import abandonment_chain, race_subgen
from most_queue.theory.utils.conv import conv_moments


class BulkServiceMM1Calc(BaseQueue):
    """
    Exact M/M^[a,b]/1 bulk-service queue via a truncated CTMC.

    :param a: minimum batch size to start a service (server waits until `a` are queued).
    :param b: maximum batch size.
    :param queue_truncation: cap on the number waiting (state-space bound).
    """

    def __init__(self, a: int = 1, b: int = 1, queue_truncation: int = 300, gamma: float = 0.0):
        super().__init__(n=1)
        if not 1 <= a <= b:
            raise ValueError("require 1 <= a <= b")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.a = a
        self.b = b
        self.N = queue_truncation
        self.gamma = gamma
        self.l = None
        self.mu_fn: Callable[[int], float] | None = None
        self.boundary_mass = None
        self._pi: np.ndarray | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu):  # pylint: disable=arguments-differ
        """
        :param mu: batch-service rate. A scalar (batch-size-independent, Exp(mu)),
            or a callable mu(batch_size) -> rate (e.g. LLM batching where a bigger
            batch is slower per batch but amortises across requests).
        """
        if callable(mu):
            self.mu_fn = mu
        else:
            self.mu_fn = lambda _i, _mu=float(mu): _mu
        self.is_servers_set = True

    def _index(self, i, j):
        return i * (self.N + 1) + j

    def _solve_pi(self) -> np.ndarray:
        """Build and solve the CTMC; cache and return the stationary distribution pi."""
        if self._pi is not None:
            return self._pi
        self._check_if_servers_and_sources_set()

        a, b, N, lam, gamma = self.a, self.b, self.N, self.l, self.gamma
        n_states = (b + 1) * (N + 1)
        rows, cols, vals = [], [], []

        def add(src, dst, rate):
            rows.append(src)
            cols.append(dst)
            vals.append(rate)

        for i in range(b + 1):
            for j in range(N + 1):
                s = self._index(i, j)
                # arrival
                if i == 0:  # idle: accumulate until `a` waiting, then start a batch
                    if j + 1 < a:
                        add(s, self._index(0, j + 1), lam)
                    else:  # j + 1 == a -> start a batch of size a
                        add(s, self._index(a, 0), lam)
                else:  # busy: arrival queues
                    if j < N:
                        add(s, self._index(i, j + 1), lam)
                # batch-service completion
                if i >= 1:
                    rate = self.mu_fn(i)
                    if j >= a:
                        take = min(b, j)
                        add(s, self._index(take, j - take), rate)
                    else:
                        add(s, self._index(0, j), rate)
                # abandonment: each of the j waiting customers independently reneges
                if j >= 1 and gamma > 0:
                    add(s, self._index(i, j - 1), j * gamma)

        Q = sp.coo_matrix((vals, (rows, cols)), shape=(n_states, n_states)).tocsr()
        out = np.asarray(Q.sum(axis=1)).ravel()
        Q = Q - sp.diags(out)

        # stationary distribution: pi Q = 0, sum pi = 1
        A = Q.transpose().tolil()
        A[0, :] = 1.0
        rhs = np.zeros(n_states)
        rhs[0] = 1.0
        pi = spla.spsolve(A.tocsr(), rhs)
        pi = np.maximum(np.real(pi), 0.0)
        pi = pi / pi.sum()
        self._pi = pi
        return pi

    def get_n_moments(self, num: int = 4) -> list[float]:
        """Exact raw moments of N (number in system), any a, b -- direct summation over pi."""
        pi = self._solve_pi()
        ig, jg = np.divmod(np.arange(len(pi)), self.N + 1)
        n = ig + jg
        return [float((pi * n**k).sum()) for k in range(1, num + 1)]

    def _abandon_chain_for_state(self, rate_first: float, j: int, start_idle: bool = False):
        """Build (cache-free) the EPIC-068 abandonment chain for a given state (MM1: k=1)."""
        b, lam, gamma, mu = self.b, self.l, self.gamma, self.mu_fn
        n_first = 0 if start_idle else 1
        k = 0 if start_idle else 1
        return abandonment_chain(rate_first, n_first, mu(b), k, j, self.a, b, lam, gamma, start_idle=start_idle)

    def get_abandonment_prob(self) -> float:
        """
        EPIC-068: exact probability that a PASTA-arriving (tagged) customer
        abandons (Markovian patience, rate ``gamma``, same convention as
        ``most_queue.theory.impatience.mm1.MM1Impatience``) before their own
        batch starts service. 0.0 when ``gamma == 0``.

        Ahead-of-tagged customers can ALSO abandon while the tagged customer
        waits -- this is NOT a simple competing-exponential add-on to the
        EPIC-067 chain (confirmed numerically wrong, ~5-30% error depending
        on state); see ``most_queue.theory.batch._idle_refill.abandonment_chain``
        for the correct construction (the ahead-of-tagged count becomes an
        explicit state dimension, a pure-death process racing against batch
        formation and new arrivals).
        """
        if self.gamma <= 0:
            return 0.0
        pi = self._solve_pi()
        a, mu = self.a, self.mu_fn
        cache: dict[tuple, float] = {}
        p_abandon = 0.0
        for idx, p in enumerate(pi):
            if p <= 0:
                continue
            i, j = divmod(idx, self.N + 1)
            if i == 0:
                if j == a - 1:
                    continue  # W=0 deterministically, can't abandon
                key = ("idle", j)
                cached = cache.get(key)
                if cached is None:
                    subgen, m, start, _ = self._abandon_chain_for_state(0.0, j, start_idle=True)
                    alpha = np.zeros(m)
                    alpha[start] = 1.0
                    x = spla.spsolve((-subgen).tocsc(), np.ones(m))
                    cached = self.gamma * float(alpha @ x)
                    cache[key] = cached
            else:
                key = ("busy", mu(i), j)
                cached = cache.get(key)
                if cached is None:
                    subgen, m, start, _ = self._abandon_chain_for_state(mu(i), j)
                    alpha = np.zeros(m)
                    alpha[start] = 1.0
                    x = spla.spsolve((-subgen).tocsc(), np.ones(m))
                    cached = self.gamma * float(alpha @ x)
                    cache[key] = cached
            p_abandon += p * cached
        return p_abandon

    def get_w(self, num: int = 4) -> list[float]:
        """
        Exact raw moments of W (waiting time) CONDITIONAL on being served, any
        ``1 <= a <= b``. At ``gamma == 0`` this is the unconditional W (see
        module docstring for the idle-refill decomposition); at ``gamma > 0``
        (EPIC-068) a tagged customer who abandons never contributes a "wait"
        in the usual sense, so these moments are normalized by
        ``1 - get_abandonment_prob()`` -- use ``get_abandonment_prob()``
        alongside this to recover the full picture.
        """
        if self.gamma > 0:
            return self._get_w_with_abandonment(num)
        pi = self._solve_pi()
        a, b, lam, mu = self.a, self.b, self.l, self.mu_fn

        def exp_moments(rate: float) -> list[float]:
            return [math.factorial(k) / rate**k for k in range(1, num + 1)]

        def erlang_moments(n_terms: int, rate: float) -> list[float]:
            if n_terms <= 0:
                return [0.0] * num
            acc = exp_moments(rate)
            for _ in range(n_terms - 1):
                acc = list(conv_moments(acc, exp_moments(rate), num))
            return acc

        moment_cache: dict[tuple, list[float]] = {}

        def race_moments(rate_cur: float, full_ahead: int, need: int) -> list[float]:
            key = (rate_cur, full_ahead, need)
            cached = moment_cache.get(key)
            if cached is not None:
                return cached
            service_rates = np.full(1 + full_ahead, mu(b))
            service_rates[0] = rate_cur
            subgen, m, start = race_subgen(service_rates, lam, need)
            alpha = np.zeros(m)
            alpha[start] = 1.0
            neg_a = (-subgen).tocsc()
            x = np.ones(m)
            moments = []
            for k in range(1, num + 1):
                x = spla.spsolve(neg_a, x)
                moments.append(math.factorial(k) * float(alpha @ x))
            moment_cache[key] = moments
            return moments

        w_moments = np.zeros(num)
        for idx, p in enumerate(pi):
            if p <= 0:
                continue
            i, j = divmod(idx, self.N + 1)
            if i == 0:
                wm = erlang_moments(a - j - 1, lam)  # idle-refill wait, if any
            else:
                full_ahead = j // b
                remainder = j - full_ahead * b
                need = a - remainder - 1
                if need > 0:  # remainder + tagged customer not enough to start a batch yet
                    wm = race_moments(mu(i), full_ahead, need)
                else:
                    wm = exp_moments(mu(i))
                    if full_ahead > 0:
                        wm = list(conv_moments(wm, erlang_moments(full_ahead, mu(b)), num))
            w_moments += p * np.array(wm)
        return list(w_moments)

    def _get_w_with_abandonment(self, num: int = 4) -> list[float]:
        """EPIC-068: exact raw moments of W conditional on being served, gamma > 0."""
        pi = self._solve_pi()
        a, mu = self.a, self.mu_fn
        cache: dict[tuple, list[float]] = {}
        numerator = np.zeros(num)
        p_served_total = 0.0
        for idx, p in enumerate(pi):
            if p <= 0:
                continue
            i, j = divmod(idx, self.N + 1)
            if i == 0 and j == a - 1:
                p_served_total += p  # W=0 deterministically, no abandonment possible
                continue
            key = ("idle", j) if i == 0 else ("busy", mu(i), j)
            cached = cache.get(key)
            if cached is None:
                if i == 0:
                    subgen, m, start, serve_rate = self._abandon_chain_for_state(0.0, j, start_idle=True)
                else:
                    subgen, m, start, serve_rate = self._abandon_chain_for_state(mu(i), j)
                alpha = np.zeros(m)
                alpha[start] = 1.0
                neg_a = (-subgen).tocsc()
                # E[W^n; served] = n! * alpha @ (-A)^-(n+1) @ serve_rate -- one MORE
                # application of (-A)^-1 than the gamma==0 "time to absorption" moment
                # formula (n! * alpha @ (-A)^-n @ 1) needs, since serve_rate is itself a
                # rate vector (not the terminal "1" reward); a first version omitted this
                # extra application and silently computed alpha @ (-A)^-1 @ serve_rate ==
                # p_served_state for every moment -- caught via a full-system DES
                # cross-check (get_w() off by ~2x at a=1) before shipping.
                x = spla.spsolve(neg_a, serve_rate)
                moments = []
                for n in range(1, num + 1):
                    x = spla.spsolve(neg_a, x)
                    moments.append(math.factorial(n) * float(alpha @ x))
                p_served_state = 1.0 - self.gamma * float(alpha @ spla.spsolve(neg_a, np.ones(m)))
                cached = (moments, p_served_state)
                cache[key] = cached
            moments, p_served_state = cached
            numerator += p * np.array(moments)
            p_served_total += p * p_served_state
        if p_served_total <= 0:
            return [0.0] * num
        return list(numerator / p_served_total)

    def get_tail(self, t: float) -> float:
        """
        Exact P(W > t), any ``1 <= a <= b``.

        Each state decomposes W into a sequential generalized-Erlang
        phase-type chain: [remaining current batch, rate mu(i), only if
        busy] + [j // b phases at rate mu(b), full batches ahead] +
        [idle-refill phase(s) at rate lambda, only if the remainder after
        the full batches ahead -- plus the tagged customer -- is still
        below the threshold a; see module docstring]. The tail of that
        chain is computed via the *sparse* matrix-exponential-action of its
        (bidiagonal, state-specific) sub-generator (``expm_multiply``, not a
        dense ``expm``: the latter is O(m^3) and, empirically, orders of
        magnitude slower here for the larger chains a big
        ``queue_truncation`` can produce) -- exact, with no hand
        partial-fraction case analysis for repeated rates.
        """
        if t < 0:
            raise ValueError("t must be nonnegative")
        if self.gamma > 0:
            return self._get_tail_with_abandonment(t)
        pi = self._solve_pi()
        a, b, lam, mu = self.a, self.b, self.l, self.mu_fn
        cache: dict[tuple, float] = {}

        def chain_tail(subgen, m, start) -> float:
            return float(spla.expm_multiply(subgen * t, np.ones(m))[start])

        tail = 0.0
        for idx, p in enumerate(pi):
            if p <= 0:
                continue
            i, j = divmod(idx, self.N + 1)
            if i == 0:
                need = a - j - 1
                if need > 0:
                    key = ("idle", need)
                    cached = cache.get(key)
                    if cached is None:
                        rates = np.full(need, lam)
                        subgen = sp.diags([-rates, rates[:-1]], [0, 1], format="csc")
                        cached = float(spla.expm_multiply(subgen * t, np.ones(need))[0])
                        cache[key] = cached
                    tail += p * cached
                continue
            full_ahead = j // b
            remainder = j - full_ahead * b
            need = a - remainder - 1
            key = (mu(i), full_ahead, max(need, 0))
            cached = cache.get(key)
            if cached is None:
                if need > 0:
                    service_rates = np.full(1 + full_ahead, mu(b))
                    service_rates[0] = mu(i)
                    subgen, m, start = race_subgen(service_rates, lam, need)
                    cached = chain_tail(subgen, m, start)
                else:
                    m = 1 + full_ahead
                    rates = np.full(m, mu(b))
                    rates[0] = mu(i)
                    subgen = sp.diags([-rates, rates[:-1]], [0, 1], format="csc")
                    cached = float(spla.expm_multiply(subgen * t, np.ones(m))[0])
                cache[key] = cached
            tail += p * cached
        return tail

    def _get_tail_with_abandonment(self, t: float) -> float:
        """EPIC-068: exact P(W > t AND served) / P(served), gamma > 0."""
        pi = self._solve_pi()
        a, mu = self.a, self.mu_fn
        cache: dict[tuple, tuple] = {}
        numerator = 0.0
        p_served_total = 0.0
        for idx, p in enumerate(pi):
            if p <= 0:
                continue
            i, j = divmod(idx, self.N + 1)
            if i == 0 and j == a - 1:
                p_served_total += p
                continue
            key = ("idle", j) if i == 0 else ("busy", mu(i), j)
            cached = cache.get(key)
            if cached is None:
                if i == 0:
                    subgen, m, start, serve_rate = self._abandon_chain_for_state(0.0, j, start_idle=True)
                else:
                    subgen, m, start, serve_rate = self._abandon_chain_for_state(mu(i), j)
                alpha = np.zeros(m)
                alpha[start] = 1.0
                neg_a = (-subgen).tocsc()
                v = spla.spsolve(neg_a, serve_rate)  # (-A)^-1 . serve_rate
                p_served_state = 1.0 - self.gamma * float(alpha @ spla.spsolve(neg_a, np.ones(m)))
                cached = (subgen, v, start, p_served_state)
                cache[key] = cached
            subgen, v, start, p_served_state = cached
            numerator += p * float(spla.expm_multiply(subgen * t, v)[start])
            p_served_total += p * p_served_state
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
        b, N = self.b, self.N

        ig, jg = np.divmod(np.arange(len(pi)), N + 1)
        e_n = float((pi * (ig + jg)).sum())
        e_t = e_n / self.l

        w = self.get_w()

        self.boundary_mass = float(pi[jg == N].sum())

        p_n = np.zeros(N + b + 1)
        for idx, prob in enumerate(pi):
            p_n[ig[idx] + jg[idx]] += prob

        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=list(w),
            p=list(p_n),
            utilization=1.0 - float(pi[ig == 0].sum()),
        )
        self._set_duration(res, start)
        return res
