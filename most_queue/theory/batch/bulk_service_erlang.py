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

Mean-only (E[N], E[V], E[W] via Little's law), same accuracy level
EPIC-012 originally shipped for the exponential case before EPIC-032 added
exact moments as a separate follow-up -- exact moments for this
phase-type-augmented case are a reserve item (the PASTA-based tagged-
customer argument EPIC-032 uses would need to account for which Erlang
phase an arrival finds the batch in, not attempted here).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue


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
        self.rate = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, rate: float):  # pylint: disable=arguments-differ
        """:param rate: per-phase Erlang rate; total mean batch-service time = k / rate."""
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self.is_servers_set = True

    def set_servers_from_moments(self, moments: list[float]):
        """Fit (k, rate) from raw moments (mean, second moment, ...) via ErlangDistribution.get_params."""
        params: ErlangParams = ErlangDistribution.get_params(moments)
        self.k = params.r
        self.rate = params.mu
        self.is_servers_set = True

    def _idle_index(self, j: int) -> int:
        return j

    def _busy_index(self, i: int, p: int, j: int) -> int:
        return (self.N + 1) + ((i - 1) * self.k + p) * (self.N + 1) + j

    def run(self) -> QueueResults:
        """Build and solve the CTMC; return mean waiting/sojourn moments (means)."""
        self._check_if_servers_and_sources_set()
        start = self._measure_time()

        a, b, k, N, lam, rate = self.a, self.b, self.k, self.N, self.l, self.rate
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

        e_n = 0.0
        for j in range(N + 1):
            e_n += pi[self._idle_index(j)] * j
        for i in range(1, b + 1):
            for p in range(k):
                for j in range(N + 1):
                    e_n += pi[self._busy_index(i, p, j)] * (i + j)

        e_t = e_n / lam
        mean_batch_service = k / rate
        e_w = e_t - mean_batch_service

        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=[e_w, 0, 0, 0],
            utilization=1.0 - float(sum(pi[self._idle_index(j)] for j in range(N + 1))),
        )
        self._set_duration(res, start)
        return res
