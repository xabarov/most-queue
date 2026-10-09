"""
EPIC-074 (M2): occupancy-dependent TWO-BRANCH (H2-like) service with a hard
concurrency cap -- the variable-service-time companion to
``occupancy_dependent.OccupancyDependentQueueCalc``.

Motivation. In ``OccupancyDependentQueueCalc`` every resident request completes
at the same occupancy-dependent rate ``mu(occ)``, i.e. remaining generation
length is exponential. Real LLM requests are strongly heterogeneous in output
length (short answers vs long ones), which exponential service cannot express.
Here each resident request additionally carries a BRANCH index (0 or 1,
"short-ish" / "long-ish"), drawn once when the request is admitted to the
active pool, and completes at a branch- AND occupancy-dependent rate.

Naming caveat (important, stated up front). Because the rate of an
already-admitted request changes whenever occupancy changes, an individual
request's service time is NOT an H2 random variable -- it is an
occupancy-MODULATED two-branch exponential. We therefore call the model
"occupancy-modulated two-branch service", not "H2 service", and only call it
H2 in the degenerate constant-occupancy reading. (The same caveat applies to
the exponential sibling: there the service time is not Exp either, it is a
piecewise-exponential with occupancy-dependent rate -- the standard
"load-dependent service rate" semantics.)

State space. ``(j, m)``: ``j`` = total number in system (served + waiting),
``m`` = how many of the ``min(j, k)`` ACTIVE requests are currently in branch
0 (the rest are in branch 1). Requests are exchangeable, so only the COUNT
matters -- this is what keeps the per-level state count at ``min(j,k)+1``
instead of ``2**j``.

Structure (verified numerically, see tests). Transitions only ever go
``j -> j+-1`` (no level skipping). The level blocks become identical from
level ``k+1`` onward -- NOT from ``k``: at ``j == k`` the queue is empty, so a
completion drops occupancy to ``k-1`` with no refill, whereas at ``j > k`` the
freed slot is refilled instantly and occupancy stays at ``k``. Levels
``0..k`` are therefore the boundary and ``k+1, k+2, ...`` are a homogeneous
QBD. We solve the truncated generator directly (sparse linear solve) rather
than via the matrix-geometric ``R``; the homogeneity is what guarantees the
truncation converges geometrically, and the direct solve keeps the code
uniform with the rest of the library's CTMC family.

Scope / honest positioning. The solution machinery here (level-structured
CTMC, QBD) is entirely standard -- Neuts (1981), Bright & Taylor (1995),
and in the Russian tradition the Takahashi-Takami method (Ryzhikov 2018,
ch. 7). Occupancy-dependent service rates in multiserver queues go back to
Bhat (1966, GI/M/2). The contribution of this module is the MODEL (that
specific combination: occupancy-modulated two-branch service + hard
concurrency cap + external FCFS queue, as a model of LLM continuous
batching with heterogeneous output lengths) and its validated
implementation -- not a new solution method. See
``docs/диссертация/литература/источники.md`` for the full source map.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue

_NEGLIGIBLE_PROB = 1e-14


class OccupancyDependentH2QueueCalc(BaseQueue):
    """
    Occupancy-modulated two-branch service with a concurrency cap `k`.

    :param k: concurrency cap (number of slots in the active pool).
    :param queue_truncation: cap `N` on the total number in system; must be >= k.
    """

    def __init__(self, k: int, queue_truncation: int = 300):
        super().__init__(n=k)
        if queue_truncation < k:
            raise ValueError(f"queue_truncation must be >= k, got {queue_truncation} < {k}")
        self.k = k
        self.N = queue_truncation
        self.l = None
        self.p1_fn = None
        self.mu1_fn = None
        self.mu2_fn = None
        self._pi: np.ndarray | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        if l <= 0:
            raise ValueError(f"arrival rate must be positive, got {l}")
        self.l = l
        self.is_sources_set = True

    def set_servers(self, p1, mu1, mu2):  # pylint: disable=arguments-differ
        """
        :param p1: probability a newly admitted request takes branch 0. Scalar, or a
            callable ``p1(occupancy) -> value`` evaluated at the occupancy the request
            is admitted into.
        :param mu1: branch-0 per-request completion rate; scalar or ``mu1(occupancy)``.
        :param mu2: branch-1 per-request completion rate; scalar or ``mu2(occupancy)``.
        """

        def _as_fn(value, name, prob=False):
            if callable(value):
                fn = value
            else:
                constant = float(value)

                def fn(_occ, _c=constant):
                    return _c

            for occ in range(1, self.k + 1):
                v = fn(occ)
                if prob and not 0.0 <= v <= 1.0:
                    raise ValueError(f"{name}({occ}) must be in [0, 1], got {v}")
                if not prob and v <= 0:
                    raise ValueError(f"{name}({occ}) must be positive, got {v}")
            return fn

        self.p1_fn = _as_fn(p1, "p1", prob=True)
        self.mu1_fn = _as_fn(mu1, "mu1")
        self.mu2_fn = _as_fn(mu2, "mu2")
        self.is_servers_set = True

    # ---------------------------------------------------------------- state indexing
    def _busy(self, j: int) -> int:
        return min(j, self.k)

    def _level_size(self, j: int) -> int:
        """Number of microstates at level j: m = 0..busy, i.e. busy+1."""
        return self._busy(j) + 1

    def _offsets(self) -> list[int]:
        off, acc = [], 0
        for j in range(self.N + 1):
            off.append(acc)
            acc += self._level_size(j)
        off.append(acc)
        return off

    def _index(self, offsets: list[int], j: int, m: int) -> int:
        return offsets[j] + m

    def _solve_pi(self) -> np.ndarray:
        """Build the truncated generator and solve pi Q = 0, sum pi = 1."""
        if self._pi is not None:
            return self._pi
        self._check_if_servers_and_sources_set()

        k, N, lam = self.k, self.N, self.l
        p1_fn, mu1_fn, mu2_fn = self.p1_fn, self.mu1_fn, self.mu2_fn
        offsets = self._offsets()
        n_states = offsets[N + 1]
        rows, cols, vals = [], [], []

        def add(src, dst, rate):
            rows.append(src)
            cols.append(dst)
            vals.append(rate)

        for j in range(N + 1):
            busy = self._busy(j)
            for m in range(busy + 1):
                s = self._index(offsets, j, m)
                # ---- arrival ----
                if j < N:
                    if j < k:
                        # newcomer is admitted immediately, draws a branch at occupancy j+1
                        pp = p1_fn(j + 1)
                        add(s, self._index(offsets, j + 1, m + 1), lam * pp)
                        add(s, self._index(offsets, j + 1, m), lam * (1.0 - pp))
                    else:
                        # cap full: newcomer queues, active composition unchanged
                        add(s, self._index(offsets, j + 1, m), lam)
                # ---- completions ----
                if busy > 0:
                    r0 = m * mu1_fn(busy)
                    r1 = (busy - m) * mu2_fn(busy)
                    if j > k:
                        # freed slot refilled instantly from the queue; occupancy stays k,
                        # the entering request draws a fresh branch at occupancy k
                        pp = p1_fn(k)
                        if r0 > 0:
                            add(s, self._index(offsets, j - 1, m), r0 * pp)
                            add(s, self._index(offsets, j - 1, m - 1), r0 * (1.0 - pp))
                        if r1 > 0:
                            add(s, self._index(offsets, j - 1, m + 1), r1 * pp)
                            add(s, self._index(offsets, j - 1, m), r1 * (1.0 - pp))
                    else:
                        # no refill: occupancy drops to j-1
                        if r0 > 0:
                            add(s, self._index(offsets, j - 1, m - 1), r0)
                        if r1 > 0:
                            add(s, self._index(offsets, j - 1, m), r1)

        q = sp.coo_matrix((vals, (rows, cols)), shape=(n_states, n_states)).tocsr()
        out = np.asarray(q.sum(axis=1)).ravel()
        q = q - sp.diags(out)

        a = q.transpose().tolil()
        a[0, :] = 1.0
        rhs = np.zeros(n_states)
        rhs[0] = 1.0
        pi = spla.spsolve(a.tocsr(), rhs)
        pi = np.maximum(np.real(pi), 0.0)
        pi /= pi.sum()
        self._pi = pi
        return pi

    # ---------------------------------------------------------------- metrics
    def get_level_probs(self) -> list[float]:
        """Marginal distribution of the total number in system (summed over branches)."""
        pi = self._solve_pi()
        offsets = self._offsets()
        return [float(pi[offsets[j] : offsets[j + 1]].sum()) for j in range(self.N + 1)]

    def get_n_moments(self, num: int = 4) -> list[float]:
        """Exact raw moments of N (number in system)."""
        p = np.asarray(self.get_level_probs())
        j = np.arange(len(p))
        return [float((p * j**r).sum()) for r in range(1, num + 1)]

    def get_p_wait(self) -> float:
        """Exact probability that an arriving (PASTA) request finds the cap full and must wait."""
        p = self.get_level_probs()
        return float(sum(p[self.k :]))

    def _wait_phase_generator(self):
        """
        Absorbing chain for the wait of a tagged request that arrives to find
        `j >= k`. State: (remaining ahead-of-tagged count r = 1..j-k+1, active
        branch composition m = 0..k). Absorption = the tagged request is admitted.

        Unlike the exponential sibling, the departure rate above the cap is NOT a
        constant: it depends on the current branch composition, which itself keeps
        changing as requests complete and are replaced. So the wait is a genuine
        phase-type distribution rather than a plain Erlang -- this is exactly what
        the two-branch extension costs (and why the exponential case was so simple).
        """
        raise NotImplementedError(
            "Phase-type waiting-time distribution for the two-branch case is a "
            "documented reserve (see module docstring and EPIC-074): the "
            "above-cap departure rate is composition-dependent, so the wait is "
            "not a plain Erlang mixture. Use get_w_mean() (exact, via Little's "
            "law) meanwhile."
        )

    def get_w_mean(self) -> float:
        """
        Exact mean waiting time before admission into the active pool, via
        Little's law applied to the QUEUE (requests present beyond the cap):
        E[W] = E[L_queue] / lambda, with E[L_queue] = sum_j (j-k)^+ pi_j.
        """
        p = np.asarray(self.get_level_probs())
        j = np.arange(len(p))
        lq = float((p * np.maximum(j - self.k, 0)).sum())
        return lq / self.l

    def get_utilization(self) -> float:
        """Mean fraction of the k slots that are occupied."""
        p = np.asarray(self.get_level_probs())
        j = np.arange(len(p))
        return float((p * np.minimum(j, self.k)).sum()) / self.k

    def run(self, num_of_moments: int = 1) -> QueueResults:  # pylint: disable=unused-argument
        """
        Run the full calculation. Only the FIRST moment of W is produced -- the
        full waiting-time distribution is an explicit reserve for this model
        (see ``_wait_phase_generator``), unlike the exponential sibling.
        """
        start = self._measure_time()
        p = self.get_level_probs()
        w = [self.get_w_mean()]
        result = QueueResults(p=p, w=w, utilization=self.get_utilization())
        self._set_duration(result, start)
        return result
