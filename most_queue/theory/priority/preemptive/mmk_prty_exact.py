"""
Exact solver for M/M/k with m preemptive-resume priority classes.

Builds the full continuous-time Markov chain on the class-count vector
(n_1, ..., n_m) truncated at a per-class cap, and solves for the stationary
distribution directly. Exact (up to truncation) for exponential service with
class-dependent rates and any number of servers/classes.

This is the reference model used to validate the RDR / RDR-A approximations
without simulation noise. It is O(prod_i (N_i + 1)) in memory, so it is meant
for small m and moderate truncation, not production throughput.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.structs import PriorityResults
from most_queue.theory.base_queue import BaseQueue


class MMkPriorityExact(BaseQueue):
    """
    Exact truncated-CTMC solver for M/M/k, m preemptive-resume priority classes.

    Classes are given highest priority first. Under preemptive-resume priority a
    class-i job occupies a server only if fewer than k higher-or-equal priority
    jobs are present; the number of class-i jobs in service is
    s_i = min(n_i, max(0, k - sum_{j<i} n_j)).
    """

    # Upper bound on the truncated state space, to keep the power-iteration solve feasible.
    MAX_STATES = 1_500_000

    def __init__(
        self,
        n: int,
        truncation: int | list[int] | None = None,
        tol: float = 1e-11,
        max_iter: int = 200_000,
        with_variance: bool = False,
    ):
        """
        :param n: number of servers.
        :param truncation: per-class state-space cap N_i. Scalar (same cap for
            all classes), list (one per class), or None (auto-sized from load).
        :param tol: convergence tolerance for the power iteration.
        :param max_iter: maximum power-iteration steps.
        :param with_variance: also compute the exact per-class response-time
            second raw moment (tagged-job method, paper's §2.4). Adds two sparse
            linear solves per class.
        """
        super().__init__(n=n)
        self.n = n
        self.truncation = truncation
        self.tol = tol
        self.max_iter = max_iter
        self.with_variance = with_variance
        self.lambdas: list[float] = []
        self.mus: list[float] = []
        self.num_classes = 0
        self.boundary_mass = None
        self.n_iter_ = 0
        self.tagged_mean_check: list | None = None

    def set_sources(self, class_arrival_rates: list[float]):  # pylint: disable=arguments-differ
        """:param class_arrival_rates: arrival rate of each class, highest priority first."""
        self.lambdas = list(class_arrival_rates)
        self.num_classes = len(self.lambdas)
        self.is_sources_set = True

    def set_servers(self, class_service_rates: list[float]):  # pylint: disable=arguments-differ
        """:param class_service_rates: service rate (mu) of each class, highest priority first."""
        self.mus = list(class_service_rates)
        self.is_servers_set = True

    def _auto_truncation(self) -> list[int]:
        """Size each class cap from the load it and its lower-priority peers see."""
        caps = []
        for i in range(self.num_classes):
            # cumulative load of classes with priority >= i (they share the k servers first)
            rho_i = sum(self.lambdas[j] / self.mus[j] for j in range(i + 1)) / self.n
            rho_i = min(rho_i, 0.97)
            # heuristic: mean queue ~ rho/(1-rho); keep several times that, with a
            # floor/ceiling. Response-time variance has a heavier tail than the
            # mean, so widen the caps when it is requested.
            factor = 9.0 if self.with_variance else 5.0
            ceiling = 140 if self.with_variance else 80
            cap = int(min(ceiling, max(12, factor / (1.0 - rho_i))))
            caps.append(cap)
        # shrink uniformly if the product would blow past the state budget
        while np.prod([c + 1 for c in caps]) > self.MAX_STATES and max(caps) > 6:
            j = int(np.argmax(caps))
            caps[j] -= 1
        return caps

    def _response_second_moment(self, i: int, pi, counts, caps) -> tuple[float, float]:
        """
        Exact second raw moment of class-i response time by the tagged-job method
        (paper's §2.4). A class-i arrival, by PASTA, sees the stationary state; its
        response time is the absorption time of an auxiliary chain whose state is
        (counts of the higher classes 0..i-1, tag position `a`), where `a` is the
        number of class-i jobs at or ahead of the tag (the tag is served last of
        those present at arrival, FCFS within class). New class-i arrivals go
        behind the tag and are dropped; lower classes never delay class i and are
        dropped. Absorption = the tag completes service.

        Moments of the absorption time solve (-T) m1 = 1, (-T) m2 = 2 m1, where T
        is the transient generator; the answer is E[m2] over the PASTA-weighted
        initial states.
        """
        lam = np.asarray(self.lambdas, dtype=float)
        mu = np.asarray(self.mus, dtype=float)
        k = self.n

        # sub-state = (n_0, ..., n_{i-1}, t) with t = a - 1 in 0..caps[i]
        sub_sizes = [caps[j] + 1 for j in range(i)] + [caps[i] + 1]
        d = len(sub_sizes)
        sub_total = int(np.prod(sub_sizes))
        strides = np.ones(d, dtype=np.int64)
        for a_ in range(d - 2, -1, -1):
            strides[a_] = strides[a_ + 1] * sub_sizes[a_ + 1]

        grids = np.meshgrid(*[np.arange(s) for s in sub_sizes], indexing="ij")
        sub = np.stack([g.ravel() for g in grids], axis=1)  # last column = t
        idx = np.arange(sub_total, dtype=np.int64)

        higher = sub[:, :i].sum(axis=1) if i > 0 else np.zeros(sub_total, dtype=np.int64)
        a = sub[:, i] + 1  # tag position (>= 1)
        avail = np.maximum(0, k - higher)  # servers free for class i
        ahead_in_service = np.minimum(a - 1, avail)
        tag_in_service = (a <= avail).astype(float)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        total_out = np.zeros(sub_total)

        # higher-class (0..i-1) arrivals and priority-served departures
        cum = np.zeros(sub_total, dtype=np.int64)
        for j in range(i):
            remaining = np.maximum(0, k - cum)
            s_j = np.minimum(sub[:, j], remaining)
            cum = cum + sub[:, j]
            can = sub[:, j] < caps[j]
            src = idx[can]
            rows.append(src)
            cols.append(src + strides[j])
            vals.append(np.full(src.shape[0], lam[j]))
            total_out[can] += lam[j]
            rate = s_j * mu[j]
            can = rate > 0
            src = idx[can]
            rows.append(src)
            cols.append(src - strides[j])
            vals.append(rate[can])
            total_out[can] += rate[can]

        # a job ahead of the tag completes: a -> a-1 (t -> t-1)
        rate = ahead_in_service * mu[i]
        can = rate > 0
        src = idx[can]
        rows.append(src)
        cols.append(src - strides[i])
        vals.append(rate[can])
        total_out[can] += rate[can]

        # the tag itself completes -> absorption (leaves the transient set)
        total_out += tag_in_service * mu[i]

        offdiag = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(sub_total, sub_total),
        ).tocsr()
        neg_t = (sp.diags(total_out) - offdiag).tocsc()

        m1 = spla.spsolve(neg_t, np.ones(sub_total))
        m2 = spla.spsolve(neg_t, 2.0 * m1)

        # PASTA-weighted initial states: map each full state to its sub-state
        sub_idx_full = counts[:, i] * strides[i]
        for j in range(i):
            sub_idx_full = sub_idx_full + counts[:, j] * strides[j]
        w = np.bincount(sub_idx_full, weights=pi, minlength=sub_total)
        w = w / w.sum()
        return float((w * m1).sum()), float((w * m2).sum())

    def run(self) -> PriorityResults:
        """Build and solve the truncated CTMC, then apply Little's law per class."""
        self._check_if_servers_and_sources_set()
        if len(self.lambdas) != len(self.mus):
            raise ValueError("class_arrival_rates and class_service_rates must have equal length")

        start = time.process_time()
        m = self.num_classes

        if self.truncation is None:
            caps = self._auto_truncation()
        elif isinstance(self.truncation, int):
            caps = [self.truncation] * m
        else:
            caps = list(self.truncation)

        sizes = [c + 1 for c in caps]  # states per class dimension: 0..cap
        total = int(np.prod(sizes))
        if total > self.MAX_STATES:
            raise ValueError(
                f"truncated state space is {total} > {self.MAX_STATES}; "
                f"reduce the `truncation` cap or the number of classes"
            )

        # mixed-radix strides for indexing state (n_1,...,n_m)
        strides = np.ones(m, dtype=np.int64)
        for i in range(m - 2, -1, -1):
            strides[i] = strides[i + 1] * sizes[i + 1]

        lam = np.asarray(self.lambdas, dtype=float)
        mu = np.asarray(self.mus, dtype=float)
        n_servers = self.n

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        # enumerate all states as a grid of counts, vectorized per class-transition
        grids = np.meshgrid(*[np.arange(s) for s in sizes], indexing="ij")
        counts = np.stack([g.ravel() for g in grids], axis=1)  # (total, m)
        idx_all = np.arange(total, dtype=np.int64)

        # servers in service per class: s_i = min(n_i, max(0, k - sum_{j<i} n_j))
        cum_higher = np.zeros(total, dtype=np.int64)
        s_in_service = np.zeros((total, m), dtype=np.int64)
        for i in range(m):
            remaining = np.maximum(0, n_servers - cum_higher)
            s_in_service[:, i] = np.minimum(counts[:, i], remaining)
            cum_higher = cum_higher + counts[:, i]

        for i in range(m):
            # arrival of class i: n_i -> n_i + 1 (if below cap)
            can_arrive = counts[:, i] < caps[i]
            src = idx_all[can_arrive]
            dst = src + strides[i]
            rows.append(src)
            cols.append(dst)
            vals.append(np.full(src.shape[0], lam[i]))

            # departure of class i: n_i -> n_i - 1 at rate s_i * mu_i
            rate = s_in_service[:, i] * mu[i]
            can_depart = rate > 0
            src = idx_all[can_depart]
            dst = src - strides[i]
            rows.append(src)
            cols.append(dst)
            vals.append(rate[can_depart])

        row = np.concatenate(rows)
        col = np.concatenate(cols)
        val = np.concatenate(vals)

        # off-diagonal rate matrix and total out-rate per state
        R = sp.coo_matrix((val, (row, col)), shape=(total, total)).tocsr()
        out_rate = np.asarray(R.sum(axis=1)).ravel()

        # Uniformized DTMC P = I + Q/Lambda (row-stochastic); its stationary vector
        # equals the CTMC stationary vector. Solve by power iteration on P^T
        # (memory-light, O(nnz) per step) — the direct sparse LU fills in badly on
        # the multi-dimensional lattice.
        lam_unif = out_rate.max() * 1.0000001
        Pt = (R.transpose().tocsr()) / lam_unif  # P^T without the diagonal
        stay = 1.0 - out_rate / lam_unif  # diagonal of P (self-loop probability)

        pi = np.full(total, 1.0 / total)
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            pi_next = Pt.dot(pi) + stay * pi
            pi_next /= pi_next.sum()
            self.n_iter_ += 1
            if np.abs(pi_next - pi).sum() < self.tol:
                pi = pi_next
                break
            pi = pi_next
        pi = np.maximum(pi, 0.0)
        pi = pi / pi.sum()

        # boundary mass (states at any class cap) — truncation quality indicator
        at_boundary = np.zeros(total, dtype=bool)
        for i in range(m):
            at_boundary |= counts[:, i] == caps[i]
        self.boundary_mass = float(pi[at_boundary].sum())

        # per-class mean number in system -> Little's law
        v_moments: list[list[float]] = []
        w_moments: list[list[float]] = []
        for i in range(m):
            e_n = float((pi * counts[:, i]).sum())
            e_t = e_n / lam[i]
            e_w = e_t - 1.0 / mu[i]
            v_moments.append([e_t, 0, 0, 0])
            w_moments.append([e_w, 0, 0, 0])

        if self.with_variance:
            self.tagged_mean_check = []
            for i in range(m):
                e_t1, e_t2 = self._response_second_moment(i, pi, counts, caps)
                v_moments[i][1] = e_t2
                # the tagged-job mean must equal Little's-law mean (a discipline-
                # invariant consistency check on the absorbing-chain construction)
                self.tagged_mean_check.append((float(v_moments[i][0]), e_t1))

        utilization = sum(lam[i] / mu[i] for i in range(m)) / self.n
        results = PriorityResults(v=v_moments, w=w_moments, p=[], utilization=utilization)
        results.duration = time.process_time() - start
        return results
