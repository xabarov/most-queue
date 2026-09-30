"""
M/H2(p1,mu1,mu2)^[a,b]/1 bulk-service queue: general (not exponential)
batch-service time, fitted to an H2 (hyperexponential, 2-phase mixture)
phase-type representation -- the CV>=1 complement to EPIC-035's Erlang
(CV<=1) case. See docs/research/bulk-service-general-h2-2026.md for why H2
was picked as the natural EPIC-035 continuation instead of the classical
embedded-chain/PGF route (Neuts 1975, Chaudhry-Templeton).

Unlike Erlang (a sequential chain of k phases), H2 is a BRANCH: at the
moment a batch starts service, one of two exponential phases is chosen
once (phase 0 w.p. p1, rate mu1; phase 1 w.p. p2=1-p1, rate mu2) and the
batch stays in that phase until it completes -- no mid-service phase
transitions. Every "a batch starts" transition (idle-threshold reached, or
a completing batch immediately starting the next one) therefore SPLITS
into two weighted sub-transitions (rate x p1 into phase 0, rate x p2 into
phase 1), instead of Erlang's single deterministic "always start in phase
0" transition.

State: (i, phase, j) -- i = batch size in service (0 = idle), phase in
{0,1} = which H2 branch the current batch chose at its start (meaningful
only when i>=1), j = number waiting.
p1=1 (H2 collapses to Exp(mu1)) reduces EXACTLY to BulkServiceMM1Calc --
verified to full float64 precision, the primary regression test.

Mean-only (E[N], E[V], E[W] via Little's law), same scope as EPIC-035's
BulkServiceErlangCalc -- exact moments for this phase-type-augmented case
are a reserve item (would need a PASTA-based tagged-customer argument that
also tracks which phase an arrival finds the batch in).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.params import H2Params
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue


class BulkServiceH2Calc(BaseQueue):
    """
    Exact M/H2(p1,mu1,mu2)^[a,b]/1 bulk-service queue via a truncated,
    phase-augmented CTMC.

    :param a: minimum batch size to start a service.
    :param b: maximum batch size.
    :param queue_truncation: cap on the number waiting (state-space bound).
    """

    def __init__(self, a: int, b: int, queue_truncation: int = 300):
        super().__init__(n=1)
        if not 1 <= a <= b:
            raise ValueError("require 1 <= a <= b")
        self.a = a
        self.b = b
        self.N = queue_truncation
        self.l = None
        self.p1 = None
        self.mu1 = None
        self.mu2 = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, p1: float, mu1: float, mu2: float):  # pylint: disable=arguments-differ
        """:param p1: probability of choosing phase 0 (rate mu1) at batch start; phase 1 (rate mu2) w.p. 1-p1."""
        if not 0.0 <= p1 <= 1.0:
            raise ValueError(f"p1 must be in [0, 1], got {p1}")
        if mu1 <= 0 or mu2 <= 0:
            raise ValueError(f"mu1, mu2 must be positive, got {mu1}, {mu2}")
        self.p1 = p1
        self.mu1 = mu1
        self.mu2 = mu2
        self.is_servers_set = True

    def set_servers_from_moments(self, moments: list[float]):
        """Fit (p1, mu1, mu2) from raw moments (mean, second moment, ...) via H2Distribution.get_params."""
        params: H2Params = H2Distribution.get_params(moments)
        self.p1 = params.p1
        self.mu1 = params.mu1
        self.mu2 = params.mu2
        self.is_servers_set = True

    def _idle_index(self, j: int) -> int:
        return j

    def _busy_index(self, i: int, phase: int, j: int) -> int:
        return (self.N + 1) + ((i - 1) * 2 + phase) * (self.N + 1) + j

    def run(self) -> QueueResults:
        """Build and solve the CTMC; return mean waiting/sojourn moments (means)."""
        self._check_if_servers_and_sources_set()
        start = self._measure_time()

        a, b, N, lam = self.a, self.b, self.N, self.l
        p1, p2, mu1, mu2 = self.p1, 1.0 - self.p1, self.mu1, self.mu2
        n_states = (N + 1) + b * 2 * (N + 1)
        rows, cols, vals = [], [], []

        def add(src, dst, r):
            rows.append(src)
            cols.append(dst)
            vals.append(r)

        for j in range(N + 1):
            s = self._idle_index(j)
            if j + 1 < a:
                add(s, self._idle_index(j + 1), lam)
            else:  # threshold reached -> batch of size a starts, split by phase
                add(s, self._busy_index(a, 0, 0), lam * p1)
                add(s, self._busy_index(a, 1, 0), lam * p2)

        for i in range(1, b + 1):
            for phase, mu in ((0, mu1), (1, mu2)):
                for j in range(N + 1):
                    s = self._busy_index(i, phase, j)
                    if j < N:
                        add(s, self._busy_index(i, phase, j + 1), lam)
                    if j >= a:
                        take = min(b, j)
                        add(s, self._busy_index(take, 0, j - take), mu * p1)
                        add(s, self._busy_index(take, 1, j - take), mu * p2)
                    else:
                        add(s, self._idle_index(j), mu)

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
            for phase in (0, 1):
                for j in range(N + 1):
                    e_n += pi[self._busy_index(i, phase, j)] * (i + j)

        e_t = e_n / lam
        mean_batch_service = p1 / mu1 + p2 / mu2
        e_w = e_t - mean_batch_service

        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=[e_w, 0, 0, 0],
            utilization=1.0 - float(sum(pi[self._idle_index(j)] for j in range(N + 1))),
        )
        self._set_duration(res, start)
        return res
