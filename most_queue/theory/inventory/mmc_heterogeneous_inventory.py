"""
M/M/c queueing-inventory system, c HETEROGENEOUS servers (mu_1, ..., mu_c
all possibly distinct), (s,S) replenishment policy, backordering or lost
sales -- the general-c generalization of EPIC-033's
MM2QueueingInventoryHeterogeneousCalc (exactly c=2), deferred three epics
(EPIC-036, EPIC-037) in favor of faster wins before being picked up here.

State-splitting for heterogeneous servers must track WHICH servers are busy,
not just how many -- at boundary level n (n < c customers, all in service,
none waiting), the busy set is one of C(c, n) subsets of {0,...,c-1}, giving
a combinatorial total boundary phase count of 2^c - 1 (times the stock-level
count). Built mechanically (per EPIC-025's canonicalize pattern) from three
small pure functions -- arrival_target, departure_targets,
subset_service_rate -- rather than a hand-written transition table, since a
hand-derived table becomes error-prone well before c=3. Combined with the
stacked-boundary-superblock QBD trick (EPIC-028/033): n=0..c-1 boundary
super-block, repeating part starts at n=c where all c servers are always
busy (any freed server instantly grabs the next queued customer, so the
repeating part needs no subset tracking, same shape as the identical-server
case).

c=2 must reduce EXACTLY (same boundary block shapes, same rates) to
MM2QueueingInventoryHeterogeneousCalc -- the primary structural regression.
mu_1=...=mu_c must reduce exactly to MMcQueueingInventoryCalc -- the
numerical regression already established for c=2 in EPIC-033, extended here
to c=3,4,5.

See docs/roadmaps/queueing_inventory_heterogeneous_servers_general_c_roadmap.md
for the full block derivation and
docs/research/queueing-inventory-heterogeneous-servers-general-c-2026.md for
the literature.
"""

from itertools import combinations
from typing import Literal

import numpy as np

from most_queue.structs import QueueingInventoryResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.matrix.qbd import QBDSolver

Policy = Literal["backorder", "lost_sales"]


def _arrival_target(subset: tuple[int, ...], c: int) -> tuple[int, ...]:
    """Lowest-index idle server joins (deterministic 'highest-priority idle first')."""
    idle = [k for k in range(c) if k not in subset]
    return tuple(sorted(subset + (min(idle),)))


def _departure_targets(subset: tuple[int, ...], mus: list[float]) -> list[tuple[int, float, tuple[int, ...]]]:
    """One (departing_server, rate, resulting_subset) per busy server."""
    return [(k, mus[k], tuple(s for s in subset if s != k)) for k in subset]


def _subset_service_rate(subset: tuple[int, ...], mus: list[float]) -> float:
    """Total outflow (diagonal) rate -- sum of the busy servers' rates."""
    return sum(mus[k] for k in subset)


class MMcQueueingInventoryHeterogeneousCalc(BaseQueue):
    """
    M/M/c queueing-inventory system with c heterogeneous servers and an
    (s, S) replenishment policy.

    :param c: number of servers (heterogeneous, possibly all-distinct rates).
    :param s_max: S -- stock level restored to after each replenishment.
    :param s: reorder point, 0 <= s < s_max. s=0 is the (0,S) special case.
    :param policy: "backorder" (arrivals wait out a stockout, never lost) or
        "lost_sales" (arrivals during a stockout are turned away).
    """

    def __init__(
        self, c: int, s_max: int, s: int = 0, policy: Policy = "backorder", calc_params: CalcParams | None = None
    ):
        super().__init__(n=c, calc_params=calc_params)
        if c < 1:
            raise ValueError(f"c must be >= 1, got {c}")
        if s_max < 1:
            raise ValueError(f"s_max must be >= 1, got {s_max}")
        if not 0 <= s < s_max:
            raise ValueError(f"reorder point s must satisfy 0 <= s < s_max, got s={s}, s_max={s_max}")
        if policy not in ("backorder", "lost_sales"):
            raise ValueError(f"policy must be 'backorder' or 'lost_sales', got {policy!r}")
        self.c = c
        self.s_max = int(s_max)
        self.s = int(s)
        self.policy: Policy = policy
        self.l = None
        self.mus: list[float] | None = None
        self.theta = None
        self._solver: QBDSolver | None = None
        self._n_offsets: list[int] | None = None  # start index of each n's block in pi0
        self._n_sizes: list[int] | None = None  # C(c,n)*m for each n

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, mus: list[float], theta: float):  # pylint: disable=arguments-differ
        """
        :param mus: service rate of each of the c servers, in assignment-priority
            order (mus[0] is preferred whenever multiple servers are idle).
        :param theta: replenishment rate (Exp(theta) lead time), triggered
            automatically whenever stock drops to the reorder point `s`.
        """
        if len(mus) != self.c:
            raise ValueError(f"expected {self.c} service rates, got {len(mus)}")
        if any(mu <= 0 for mu in mus):
            raise ValueError(f"all service rates must be positive, got {mus}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.mus = [float(mu) for mu in mus]
        self.theta = float(theta)
        self.is_servers_set = True

    # ------------------------------------------------------------ internals
    def _build_solver(self) -> QBDSolver:
        if self._solver is not None:
            return self._solver
        self._check_if_servers_and_sources_set()

        c, mus = self.c, self.mus
        m = self.s_max + 1  # phases: stock level i = 0..S
        lam, theta = self.l, self.theta
        s, s_max = self.s, self.s_max
        lost = self.policy == "lost_sales"
        mu_sum = sum(mus)

        def arrival_row(i: int) -> float:
            return 0.0 if (lost and i == 0) else lam

        def diag_block(mu_local: float) -> np.ndarray:
            diag = np.zeros((m, m))
            for i in range(s + 1):
                diag[i, s_max] += theta
            diag[0, 0] = -theta if lost else -(lam + theta)
            for i in range(1, s + 1):
                diag[i, i] = -(arrival_row(i) + mu_local + theta)
            for i in range(s + 1, m):
                diag[i, i] = -(arrival_row(i) + mu_local)
            return diag

        def down_block(mu_local: float) -> np.ndarray:
            down = np.zeros((m, m))
            for i in range(1, m):
                down[i, i - 1] = mu_local
            return down

        arrival_diag = np.diag([arrival_row(i) for i in range(m)])

        # ----------------------------------------------- repeating part (n >= c)
        a0 = arrival_diag.copy()
        a2 = down_block(mu_sum)
        a1 = diag_block(mu_sum)

        # ----------------------------------------------- boundary (n = 0..c-1)
        subsets_by_n = {n: list(combinations(range(c), n)) for n in range(c)}
        subset_index: dict[tuple[int, ...], int] = {}
        n_offsets: list[int] = []
        n_sizes: list[int] = []
        idx = 0
        for n in range(c):
            n_offsets.append(idx)
            for subset in subsets_by_n[n]:
                subset_index[subset] = idx
                idx += 1
            n_sizes.append(idx - n_offsets[-1])
        n_boundary_subsets = idx  # = 2^c - 1
        m0 = n_boundary_subsets * m

        b00 = np.zeros((m0, m0))
        for n in range(c):
            for subset in subsets_by_n[n]:
                sidx = subset_index[subset]
                lo, hi = sidx * m, (sidx + 1) * m
                mu_local = _subset_service_rate(subset, mus)
                b00[lo:hi, lo:hi] = diag_block(mu_local)

                if n < c - 1:
                    target = _arrival_target(subset, c)
                    tidx = subset_index[target]
                    tlo, thi = tidx * m, (tidx + 1) * m
                    b00[lo:hi, tlo:thi] = arrival_diag

                for _k, mu_k, target in _departure_targets(subset, mus):
                    tidx = subset_index[target]
                    tlo, thi = tidx * m, (tidx + 1) * m
                    b00[lo:hi, tlo:thi] += down_block(mu_k)

        b01 = np.zeros((m0, m))  # boundary n=c-1 -> repeating level n=c
        for subset in subsets_by_n[c - 1]:
            sidx = subset_index[subset]
            lo, hi = sidx * m, (sidx + 1) * m
            b01[lo:hi, :] = arrival_diag

        b10 = np.zeros((m, m0))  # repeating level n=c -> boundary n=c-1, split by which server finished
        full_set = tuple(range(c))
        for k, mu_k, target in _departure_targets(full_set, mus):
            tidx = subset_index[target]
            tlo, thi = tidx * m, (tidx + 1) * m
            b10[:, tlo:thi] = down_block(mu_k)

        self._solver = QBDSolver(a0, a1, a2, b00, b01, b10)
        self._solver.solve()
        self._n_offsets = n_offsets
        self._n_sizes = n_sizes
        return self._solver

    def _utilization(self) -> float:
        rho = self.l / sum(self.mus)
        if rho >= 1:
            raise ValueError(f"System is unstable: rho={rho} must be < 1 (necessary, not sufficient here)")
        return rho

    def _n_block(self, n: int) -> np.ndarray:
        """pi0 slice for boundary level n (concatenated across its C(c,n) subsets, each m states)."""
        solver = self._build_solver()
        off = self._n_offsets[n]
        size = self._n_sizes[n]
        m = self.s_max + 1
        return solver.pi0[off * m : off * m + size * m]

    def _mean_in_system(self) -> float:
        """
        sum_{n=0}^{c-1} n*P(n) + repeating-part contribution. Repeating QBD
        level k (k=0,1,...) has actual N = c+k; its total contribution to
        E[N] is sum_k (c+k)*P_k = c*p_ge_c + sum_k k*P_k. The identity
        pi1 @ (I-R)^-2 @ 1 = sum_k (k+1)*P_k (standard matrix-geometric mean
        formula) means "mean_extra" below already includes one extra
        p_ge_c beyond sum_k k*P_k, so only (c-1) extra copies of p_ge_c are
        added explicitly -- reduces to EPIC-033's "boundary_term + p_ge_2 +
        mean_extra" exactly at c=2.
        """
        solver = self._build_solver()
        m = self.s_max + 1
        c = self.c
        boundary_term = sum(n * float(self._n_block(n).sum()) for n in range(c))

        inv = np.linalg.inv(np.eye(m) - solver.r)
        p_ge_c = float(solver.pi1 @ inv @ np.ones(m))
        mean_extra = float(solver.pi1 @ inv @ inv @ np.ones(m))
        return boundary_term + (c - 1) * p_ge_c + mean_extra

    def _phase_marginal(self) -> np.ndarray:
        """Exact stock-level distribution, summed over all levels N and all busy-subsets."""
        solver = self._build_solver()
        m = self.s_max + 1
        c = self.c
        total = np.zeros(m)
        for n in range(c):
            block = self._n_block(n).reshape(self._n_sizes[n], m)
            total += block.sum(axis=0)
        inv = np.linalg.inv(np.eye(m) - solver.r)
        total += solver.pi1 @ inv
        return total

    def _effective_arrival_rate(self) -> float:
        """lambda under backorder; lambda*(1 - stockout_prob) under lost_sales."""
        if self.policy == "backorder":
            return self.l
        stockout_prob = float(self._phase_marginal()[0])
        return self.l * (1.0 - stockout_prob)

    def _mean_service_time(self) -> float:
        """
        E[S] via flow balance: sum_k P(server k busy, i>=1) = sum over busy
        servers, summed over states = sum_{n=1}^{c-1} n*P(n, i>=1) +
        c*P(N>=c, i>=1); E[S] = that sum / lambda_eff (reduces to 1/mu at
        mu_1=...=mu_c=mu, same algebra as EPIC-033's c=2 case).
        """
        solver = self._build_solver()
        m = self.s_max + 1
        c = self.c

        boundary_active = 0.0
        for n in range(1, c):
            block = self._n_block(n)
            active = float(block.reshape(self._n_sizes[n], m)[:, 1:].sum())
            boundary_active += n * active

        inv = np.linalg.inv(np.eye(m) - solver.r)
        p_ge_c_active = float(solver.pi1 @ inv @ np.ones(m)) - float(solver.pi1 @ inv[:, 0])

        return (boundary_active + c * p_ge_c_active) / self._effective_arrival_rate()

    # ------------------------------------------------------------- results
    def get_p(self, num_levels: int | None = None) -> list[float]:
        """Probabilities of the number of customers in the system (levels)."""
        self._check_if_servers_and_sources_set()
        solver = self._build_solver()
        n_levels = num_levels or self.calc_params.p_num
        c = self.c

        p = [float(self._n_block(n).sum()) for n in range(c)][:n_levels]
        if n_levels > c:
            vec = solver.pi1.copy()
            for _ in range(n_levels - c):
                p.append(float(vec.sum()))
                vec = vec @ solver.r
        self.p = p
        return self.p

    def get_v(self) -> list[float]:
        """Mean sojourn time (first moment only): E[V] = E[N] / lambda_effective (Little's law)."""
        self.v = [self._mean_in_system() / self._effective_arrival_rate()]
        return self.v

    def get_w(self) -> list[float]:
        """
        Mean waiting time: ``E[W] = E[V] - E[service]``.

        .. warning::

           Known defect, queued for fix (see EPIC-075 and
           docs/roadmaps/literature_catchup_roadmap.md). This uses
           ``E[V] - 1/mu``, which OVERSTATES the wait whenever ``c > 1`` and
           stockouts actually occur: with several servers running, one can
           consume the last stock unit while another customer is still
           mid-service, suspending that service until a replenishment, so the
           real ``E[S]`` exceeds the nominal mean service time. The error
           vanishes when stock never binds and grows with the stockout
           probability (measured ~2% in a c=2 example). ``E[V]`` and the stock
           metrics are exact and unaffected. The exact wait is already
           available in ``MMcQueueingInventoryCalc`` via its phase-type
           construction; the heterogeneous classes need their own (the tagged
           customer's wait depends on WHICH servers are busy), which is the
           next item on the catchup roadmap.
        """
        v = self.v if self.v is not None else self.get_v()
        self.w = [v[0] - self._mean_service_time()]
        return self.w

    def get_stock_distribution(self) -> list[float]:
        """P(stock level = i), i = 0..S, exact (summed over all queue lengths)."""
        return [float(x) for x in self._phase_marginal()]

    def run(self, num_levels: int | None = None) -> QueueingInventoryResults:
        """Solve the QBD and report queue + inventory metrics."""
        start = self._measure_time()
        with self._validate_state():
            utilization = self._utilization()
            p = self.get_p(num_levels)
            v = self.get_v()
            w = self.get_w()
            stock = self.get_stock_distribution()

        loss_prob = stock[0] if self.policy == "lost_sales" else 0.0

        result = QueueingInventoryResults(
            v=v,
            w=w,
            p=p,
            utilization=utilization,
            stock_distribution=stock,
            stockout_prob=stock[0],
            fill_rate=1.0 - stock[0],
            loss_prob=loss_prob,
        )
        self._set_duration(result, start)
        return result
