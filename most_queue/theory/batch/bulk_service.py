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
from most_queue.theory.batch._idle_refill import race_subgen
from most_queue.theory.utils.conv import conv_moments


class BulkServiceMM1Calc(BaseQueue):
    """
    Exact M/M^[a,b]/1 bulk-service queue via a truncated CTMC.

    :param a: minimum batch size to start a service (server waits until `a` are queued).
    :param b: maximum batch size.
    :param queue_truncation: cap on the number waiting (state-space bound).
    """

    def __init__(self, a: int = 1, b: int = 1, queue_truncation: int = 300):
        super().__init__(n=1)
        if not 1 <= a <= b:
            raise ValueError("require 1 <= a <= b")
        self.a = a
        self.b = b
        self.N = queue_truncation
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

        a, b, N, lam = self.a, self.b, self.N, self.l
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

    def get_w(self, num: int = 4) -> list[float]:
        """
        Exact raw moments of W (waiting time), any ``1 <= a <= b`` (see module
        docstring for the idle-refill decomposition).
        """
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
