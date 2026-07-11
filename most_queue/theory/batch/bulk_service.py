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
"""

import time
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue


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

    def run(self) -> QueueResults:
        """Build and solve the CTMC; return mean waiting/sojourn moments (means)."""
        self._check_if_servers_and_sources_set()
        start = time.process_time()

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

        # metrics
        ig, jg = np.divmod(np.arange(n_states), N + 1)
        e_n = float((pi * (ig + jg)).sum())
        # mean batch-service time experienced by a customer (batch-size independent
        # unless mu depends on i; report the aggregate mean service via utilization)
        self.mean_batch_service = 1.0 / self.mu_fn(min(b, max(a, 1)))
        e_t = e_n / lam
        e_w = e_t - self.mean_batch_service

        self.boundary_mass = float(pi[jg == N].sum())

        p_n = np.zeros(N + b + 1)
        for idx in range(n_states):
            p_n[ig[idx] + jg[idx]] += pi[idx]

        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=[e_w, 0, 0, 0],
            p=list(p_n),
            utilization=1.0 - float(pi[ig == 0].sum()),
        )
        res.duration = time.process_time() - start
        return res
