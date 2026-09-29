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
  moments via ``conv_moments``. **Only exact for a=1**: for a>1, if the
  remainder after the full batches is below the threshold a, the server
  goes idle and waits for more arrivals (an "idle-refill" sub-problem that
  breaks this simple decomposition) -- confirmed numerically (~18% error
  at a=2 against Monte Carlo) before restricting the scope, see
  docs/research/bulk-service-waiting-moments-2026.md. ``run()`` uses this
  exact result for a=1 and falls back to the old (approximate, documented)
  mean-only estimate for a>1.
"""

import math
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
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
        self.mean_batch_service = None
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
        Exact raw moments of W (waiting time) -- requires a == 1 (see module
        docstring for why a > 1 is not exact with this decomposition).
        """
        if self.a != 1:
            raise ValueError(
                f"get_w() is only exact for a=1 (got a={self.a}); see "
                "docs/research/bulk-service-waiting-moments-2026.md for why a>1 needs a "
                "different (not-yet-implemented) idle-refill-aware derivation. "
                "run() still reports an approximate mean for a>1."
            )
        pi = self._solve_pi()
        b, mu = self.b, self.mu_fn

        def exp_moments(rate: float) -> list[float]:
            return [math.factorial(k) / rate**k for k in range(1, num + 1)]

        def erlang_moments(n_terms: int, rate: float) -> list[float]:
            if n_terms == 0:
                return [0.0] * num
            acc = exp_moments(rate)
            for _ in range(n_terms - 1):
                acc = list(conv_moments(acc, exp_moments(rate), num))
            return acc

        w_moments = np.zeros(num)
        for idx, p in enumerate(pi):
            if p <= 0:
                continue
            i, j = divmod(idx, self.N + 1)
            if i == 0:
                wm = [0.0] * num  # a=1: idle means j=0, batch starts on this arrival
            else:
                wm = exp_moments(mu(i))
                full_ahead = j // b
                if full_ahead > 0:
                    wm = list(conv_moments(wm, erlang_moments(full_ahead, mu(b)), num))
            w_moments += p * np.array(wm)
        return list(w_moments)

    def run(self) -> QueueResults:
        """Solve the CTMC; return waiting/sojourn moments (exact for N and, at a=1, for W)."""
        start = self._measure_time()
        pi = self._solve_pi()
        a, b, N = self.a, self.b, self.N

        ig, jg = np.divmod(np.arange(len(pi)), N + 1)
        e_n = float((pi * (ig + jg)).sum())
        e_t = e_n / self.l

        if a == 1:
            w = self.get_w()
            e_w = w[0]
        else:
            # Approximate: fixed-batch-size E[S] estimate, not exact when a<b
            # (see module docstring / docs/research/bulk-service-waiting-moments-2026.md).
            self.mean_batch_service = 1.0 / self.mu_fn(min(b, max(a, 1)))
            e_w = e_t - self.mean_batch_service
            w = [e_w, 0, 0, 0]

        self.boundary_mass = float(pi[jg == N].sum())

        p_n = np.zeros(N + b + 1)
        for idx in range(len(pi)):
            p_n[ig[idx] + jg[idx]] += pi[idx]

        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=list(w),
            p=list(p_n),
            utilization=1.0 - float(pi[ig == 0].sum()),
        )
        self._set_duration(res, start)
        return res
