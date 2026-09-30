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
record batch-START time rather than batch-COMPLETION time). Same a=1-only
restriction as EPIC-032 (see that module's docstring for why a>1 breaks
the decomposition).
"""

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
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

    def __init__(self, a: int, b: int, k: int, queue_truncation: int = 300):
        super().__init__(n=1)
        if not 1 <= a <= b:
            raise ValueError("require 1 <= a <= b")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.a = a
        self.b = b
        self.k = k
        self.N = queue_truncation
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

    def get_w(self, num: int = 4) -> list[float]:
        """
        Exact raw moments of W (waiting time) -- requires a == 1 (same
        restriction as BulkServiceMM1Calc.get_w(), see that module's
        docstring for why a > 1 breaks this decomposition).
        """
        if self.a != 1:
            raise ValueError(
                f"get_w() is only exact for a=1 (got a={self.a}); see "
                "docs/research/bulk-service-waiting-moments-2026.md for why a>1 needs a "
                "different (not-yet-implemented) idle-refill-aware derivation."
            )
        pi = self._solve_pi()
        b, k, rate_fn = self.b, self.k, self.rate_fn

        def exp_moments(rate: float) -> list[float]:
            return [math.factorial(n) / rate**n for n in range(1, num + 1)]

        def erlang_moments(n_terms: int, rate: float) -> list[float]:
            if n_terms == 0:
                return [0.0] * num
            acc = exp_moments(rate)
            for _ in range(n_terms - 1):
                acc = list(conv_moments(acc, exp_moments(rate), num))
            return acc

        w_moments = np.zeros(num)
        for i in range(1, b + 1):
            rate_i = rate_fn(i)
            for p in range(k):
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, p, j)]
                    if prob <= 0:
                        continue
                    wm = erlang_moments(k - p, rate_i)  # remaining phases of the current batch
                    full_ahead = j // b
                    if full_ahead > 0:  # each full batch ahead is always exactly size b
                        wm = list(conv_moments(wm, erlang_moments(full_ahead * k, rate_fn(b)), num))
                    w_moments += prob * np.array(wm)
        return list(w_moments)

    def run(self) -> QueueResults:
        """Solve the CTMC; return waiting/sojourn moments (exact for N and, at a=1, for W)."""
        start = self._measure_time()
        pi = self._solve_pi()
        a, b, k, N, lam, rate_fn = self.a, self.b, self.k, self.N, self.l, self.rate_fn

        e_n = 0.0
        p_busy_size = np.zeros(b + 1)  # P(batch size = i | server busy), i=1..b
        for j in range(N + 1):
            e_n += pi[self._idle_index(j)] * j
        for i in range(1, b + 1):
            for p in range(k):
                for j in range(N + 1):
                    prob = pi[self._busy_index(i, p, j)]
                    e_n += prob * (i + j)
                    p_busy_size[i] += prob

        e_t = e_n / lam

        if a == 1:
            w = self.get_w()
            e_w = w[0]
        else:
            p_busy_total = p_busy_size.sum()
            if p_busy_total > 0:
                mean_batch_service = sum((k / rate_fn(i)) * p_busy_size[i] for i in range(1, b + 1)) / p_busy_total
            else:
                mean_batch_service = k / rate_fn(b)
            e_w = e_t - mean_batch_service
            w = [e_w, 0, 0, 0]

        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=list(w),
            utilization=1.0 - float(sum(pi[self._idle_index(j)] for j in range(N + 1))),
        )
        self._set_duration(res, start)
        return res
