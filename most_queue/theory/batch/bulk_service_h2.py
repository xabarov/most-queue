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

EPIC-042 adds batch-size-dependent parameters: p1/mu1/mu2 may each be a
callable `f(batch_size) -> value`, porting BulkServiceMM1Calc's existing
callable-mu convention. A subtlety absent from the Erlang case (EPIC-042's
Erlang generalization only needed the CURRENT batch's rate): the branch
SPLIT at a "batch starts" transition must use p1 of the NEW batch about to
form, not of the batch that just completed -- e.g. at a completion event
i -> take (take = min(b, j)), the outflow rate uses the completing batch's
own (i, phase) rate, but the p1/p2 weights splitting into the new batch's
two phases must be p1(take), not p1(i). Getting this backwards would
silently use the wrong batch's H2 parameters whenever p1 varies with size.
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
        self.p1_fn = None
        self.mu1_fn = None
        self.mu2_fn = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, p1, mu1, mu2):  # pylint: disable=arguments-differ
        """
        :param p1: probability of choosing phase 0 (rate mu1) at batch start; phase 1 (rate mu2)
            w.p. 1-p1. Each of p1/mu1/mu2 may be a scalar (batch-size-independent) or a callable
            f(batch_size) -> value (e.g. LLM batching where a bigger batch has different
            variability/rate).
        """

        def _as_fn(value):
            if callable(value):
                return value
            return lambda _i, _v=float(value): _v

        self.p1_fn = _as_fn(p1)
        self.mu1_fn = _as_fn(mu1)
        self.mu2_fn = _as_fn(mu2)
        for size in range(self.a, self.b + 1):
            p1_val, mu1_val, mu2_val = self.p1_fn(size), self.mu1_fn(size), self.mu2_fn(size)
            if not 0.0 <= p1_val <= 1.0:
                raise ValueError(f"p1({size}) must be in [0, 1], got {p1_val}")
            if mu1_val <= 0 or mu2_val <= 0:
                raise ValueError(f"mu1({size}), mu2({size}) must be positive, got {mu1_val}, {mu2_val}")
        self.is_servers_set = True

    def set_servers_from_moments(self, moments: list[float]):
        """Fit (p1, mu1, mu2) from raw moments (mean, second moment, ...) via H2Distribution.get_params."""
        params: H2Params = H2Distribution.get_params(moments)
        self.set_servers(params.p1, params.mu1, params.mu2)

    def _idle_index(self, j: int) -> int:
        return j

    def _busy_index(self, i: int, phase: int, j: int) -> int:
        return (self.N + 1) + ((i - 1) * 2 + phase) * (self.N + 1) + j

    def run(self) -> QueueResults:
        """Build and solve the CTMC; return mean waiting/sojourn moments (means)."""
        self._check_if_servers_and_sources_set()
        start = self._measure_time()

        a, b, N, lam = self.a, self.b, self.N, self.l
        p1_fn, mu1_fn, mu2_fn = self.p1_fn, self.mu1_fn, self.mu2_fn
        n_states = (N + 1) + b * 2 * (N + 1)
        rows, cols, vals = [], [], []

        def add(src, dst, r):
            rows.append(src)
            cols.append(dst)
            vals.append(r)

        p1_a = p1_fn(a)
        for j in range(N + 1):
            s = self._idle_index(j)
            if j + 1 < a:
                add(s, self._idle_index(j + 1), lam)
            else:  # threshold reached -> batch of size a starts, split by phase (new batch's own p1)
                add(s, self._busy_index(a, 0, 0), lam * p1_a)
                add(s, self._busy_index(a, 1, 0), lam * (1.0 - p1_a))

        for i in range(1, b + 1):
            for phase, mu in ((0, mu1_fn(i)), (1, mu2_fn(i))):
                for j in range(N + 1):
                    s = self._busy_index(i, phase, j)
                    if j < N:
                        add(s, self._busy_index(i, phase, j + 1), lam)
                    if j >= a:
                        take = min(b, j)
                        p1_take = p1_fn(take)  # the NEW batch's own p1, not the completing batch's
                        add(s, self._busy_index(take, 0, j - take), mu * p1_take)
                        add(s, self._busy_index(take, 1, j - take), mu * (1.0 - p1_take))
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
        p_busy_size = np.zeros(b + 1)  # P(batch size = i | server busy), i=1..b
        for j in range(N + 1):
            e_n += pi[self._idle_index(j)] * j
        for i in range(1, b + 1):
            for phase in (0, 1):
                for j in range(N + 1):
                    prob = pi[self._busy_index(i, phase, j)]
                    e_n += prob * (i + j)
                    p_busy_size[i] += prob

        e_t = e_n / lam

        def mean_service_of(size):
            p1_val = p1_fn(size)
            return p1_val / mu1_fn(size) + (1.0 - p1_val) / mu2_fn(size)

        p_busy_total = p_busy_size.sum()
        if p_busy_total > 0:
            mean_batch_service = sum(mean_service_of(i) * p_busy_size[i] for i in range(1, b + 1)) / p_busy_total
        else:
            mean_batch_service = mean_service_of(b)
        e_w = e_t - mean_batch_service

        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=[e_w, 0, 0, 0],
            utilization=1.0 - float(sum(pi[self._idle_index(j)] for j in range(N + 1))),
        )
        self._set_duration(res, start)
        return res
