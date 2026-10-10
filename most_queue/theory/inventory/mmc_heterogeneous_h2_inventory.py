"""
M/H2/c queueing-inventory system, c HETEROGENEOUS servers, each with its OWN
H2(p1_k, mu1_k, mu2_k) service-time distribution (not just a scalar
exponential rate) -- combines EPIC-036's H2 phase-type CTMC augmentation
with EPIC-038's general-c heterogeneous-server subset-tracking, closing the
reserve EPIC-038 flagged and picked up after an explicit user request to
make queueing-inventory service times more realistic than exponential.

Per-server state is extended from EPIC-038's binary idle/busy flag to a
ternary one: idle (-1), or busy having chosen H2 branch 0 (rate mu1_k) or
branch 1 (rate mu2_k) at the START of the current service -- fixed until
that customer departs (H2's defining "choose once" property, unlike
Erlang's sequential-phase-advance). A "config" is a length-c tuple over
{-1, 0, 1}. Boundary phase count (n=0..c-1) is 3^c - 2^c (vs. EPIC-038's
2^c - 1); the repeating part (n>=c, all servers always busy) gets a
genuinely non-scalar 2^c-sized phase space (branch combinations across all
c servers) -- unlike EPIC-038, where the repeating part needed no phase at
all, since here completion rate depends on which branch a server is
running. H2's "choose once" property means the repeating part still has NO
within-level transitions (a busy server's branch never changes except at
departure) -- departures split into two weighted sub-transitions (redraw
the departing-then-instantly-refilled server's branch) when the queue
stays nonempty, or collapse into the boundary (no redraw) when it empties.

p1_k=1 for all k (H2 degenerates to Exp(mu1_k)) must reduce EXACTLY to
MMcQueueingInventoryHeterogeneousCalc (EPIC-038) -- the primary structural
regression, verified to full float64 precision.

Complex-valued H2 fits (fit_h2_clx) are deliberately NOT used anywhere
here: a CTMC generator's rates and branch probabilities must be real and
non-negative by construction, or the "chain" is not a valid stochastic
process. set_servers_from_moments uses H2Distribution.get_params (fit_h2,
Aliev's method), which is real by construction and degenerates gracefully
outside the H2-feasible region -- same convention as every other H2 use in
this library (EPIC-021's SLA layer, EPIC-036).

See docs/roadmaps/queueing_inventory_heterogeneous_servers_h2_service_roadmap.md
for the full block derivation and
docs/research/queueing-inventory-heterogeneous-servers-h2-service-2026.md
for the literature and the complex-H2 rationale.
"""

from itertools import product
from typing import Literal

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.params import H2Params
from most_queue.structs import QueueingInventoryResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.inventory._wait_phase import WaitPhaseTypeMixin
from most_queue.theory.matrix.qbd import QBDSolver

Policy = Literal["backorder", "lost_sales"]

Config = tuple[int, ...]  # length c, entries in {-1, 0, 1}


def _rate_of(config: Config, k: int, servers: list[H2Params]) -> float:
    return servers[k].mu1 if config[k] == 0 else servers[k].mu2


def _occupied(config: Config) -> int:
    return sum(1 for x in config if x != -1)


def _with(config: Config, k: int, value: int) -> Config:
    updated = list(config)
    updated[k] = value
    return tuple(updated)


class MMcQueueingInventoryHeterogeneousH2Calc(WaitPhaseTypeMixin, BaseQueue):
    """
    M/H2/c queueing-inventory system: c heterogeneous servers, each with its
    own H2(p1, mu1, mu2) service-time distribution, and an (s, S)
    replenishment policy.

    :param c: number of servers (heterogeneous, each with its own H2 fit).
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
        self.servers: list[H2Params] | None = None
        self.theta = None
        self._solver: QBDSolver | None = None
        self._n_offsets: list[int] | None = None
        self._n_sizes: list[int] | None = None
        self._n_rep: int | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, servers: list[H2Params], theta: float):  # pylint: disable=arguments-differ
        """
        :param servers: H2 service-time parameters of each of the c servers,
            in assignment-priority order (servers[0] preferred whenever
            multiple servers are idle).
        :param theta: replenishment rate (Exp(theta) lead time).
        """
        if len(servers) != self.c:
            raise ValueError(f"expected {self.c} server H2 params, got {len(servers)}")
        for params in servers:
            if not 0.0 <= params.p1 <= 1.0:
                raise ValueError(f"p1 must be in [0, 1], got {params.p1}")
            if params.mu1 <= 0 or params.mu2 <= 0:
                raise ValueError(f"mu1, mu2 must be positive, got {params.mu1}, {params.mu2}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.servers = list(servers)
        self.theta = float(theta)
        self.is_servers_set = True

    def set_servers_from_moments(self, moments_per_server: list[list[float]], theta: float):
        """Fit each server's (p1, mu1, mu2) independently from its own raw moments (fit_h2)."""
        if len(moments_per_server) != self.c:
            raise ValueError(f"expected {self.c} moment lists, got {len(moments_per_server)}")
        servers = [H2Distribution.get_params(moments) for moments in moments_per_server]
        self.set_servers(servers, theta)

    # ------------------------------------------------------------ internals
    def _build_solver(self) -> QBDSolver:
        if self._solver is not None:
            return self._solver
        self._check_if_servers_and_sources_set()

        c, servers = self.c, self.servers
        m = self.s_max + 1
        lam, theta = self.l, self.theta
        s, s_max = self.s, self.s_max
        lost = self.policy == "lost_sales"

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

        all_configs = list(product((-1, 0, 1), repeat=c))
        boundary_by_n = {n: [cfg for cfg in all_configs if _occupied(cfg) == n] for n in range(c)}
        repeating_configs = [cfg for cfg in all_configs if _occupied(cfg) == c]

        subset_index: dict[Config, int] = {}
        n_offsets: list[int] = []
        n_sizes: list[int] = []
        idx = 0
        for n in range(c):
            n_offsets.append(idx)
            for cfg in boundary_by_n[n]:
                subset_index[cfg] = idx
                idx += 1
            n_sizes.append(idx - n_offsets[-1])
        m0 = idx * m

        rep_index = {cfg: i for i, cfg in enumerate(repeating_configs)}
        n_rep = len(repeating_configs)  # 2^c
        big_m = n_rep * m

        # ----------------------------------------------- repeating part (n >= c)
        a0 = np.zeros((big_m, big_m))
        a1 = np.zeros((big_m, big_m))
        a2 = np.zeros((big_m, big_m))

        for cfg in repeating_configs:
            ridx = rep_index[cfg]
            lo, hi = ridx * m, (ridx + 1) * m
            mu_local = sum(_rate_of(cfg, k, servers) for k in range(c))
            a1[lo:hi, lo:hi] = diag_block(mu_local)
            a0[lo:hi, lo:hi] = arrival_diag
            for k in range(c):
                rk = _rate_of(cfg, k, servers)
                p1 = servers[k].p1
                r0, r1 = rep_index[_with(cfg, k, 0)], rep_index[_with(cfg, k, 1)]
                a2[lo:hi, r0 * m : (r0 + 1) * m] += down_block(rk * p1)
                a2[lo:hi, r1 * m : (r1 + 1) * m] += down_block(rk * (1.0 - p1))

        # ----------------------------------------------- boundary (n = 0..c-1)
        b00 = np.zeros((m0, m0))
        for n in range(c):
            for cfg in boundary_by_n[n]:
                sidx = subset_index[cfg]
                lo, hi = sidx * m, (sidx + 1) * m
                mu_local = sum(_rate_of(cfg, k, servers) for k in range(c) if cfg[k] != -1)
                b00[lo:hi, lo:hi] = diag_block(mu_local)

                if n < c - 1:
                    slot = min(k for k in range(c) if cfg[k] == -1)
                    p1 = servers[slot].p1
                    t0, t1 = subset_index[_with(cfg, slot, 0)], subset_index[_with(cfg, slot, 1)]
                    b00[lo:hi, t0 * m : (t0 + 1) * m] += arrival_diag * p1
                    b00[lo:hi, t1 * m : (t1 + 1) * m] += arrival_diag * (1.0 - p1)

                for k in range(c):
                    if cfg[k] == -1:
                        continue
                    rk = _rate_of(cfg, k, servers)
                    tidx = subset_index[_with(cfg, k, -1)]
                    b00[lo:hi, tidx * m : (tidx + 1) * m] += down_block(rk)

        b01 = np.zeros((m0, big_m))
        for cfg in boundary_by_n[c - 1]:
            sidx = subset_index[cfg]
            lo, hi = sidx * m, (sidx + 1) * m
            slot = min(k for k in range(c) if cfg[k] == -1)
            p1 = servers[slot].p1
            r0, r1 = rep_index[_with(cfg, slot, 0)], rep_index[_with(cfg, slot, 1)]
            b01[lo:hi, r0 * m : (r0 + 1) * m] += arrival_diag * p1
            b01[lo:hi, r1 * m : (r1 + 1) * m] += arrival_diag * (1.0 - p1)

        b10 = np.zeros((big_m, m0))
        for cfg in repeating_configs:
            ridx = rep_index[cfg]
            lo, hi = ridx * m, (ridx + 1) * m
            for k in range(c):
                rk = _rate_of(cfg, k, servers)
                tidx = subset_index[_with(cfg, k, -1)]
                b10[lo:hi, tidx * m : (tidx + 1) * m] += down_block(rk)

        self._solver = QBDSolver(a0, a1, a2, b00, b01, b10)
        self._solver.solve()
        self._n_offsets = n_offsets
        self._n_sizes = n_sizes
        self._n_rep = n_rep
        return self._solver

    def _utilization(self) -> float:
        """rho = lambda / sum_k(1/E[S_k]), E[S_k] = p1_k/mu1_k + (1-p1_k)/mu2_k (mean service
        time of server k's H2 distribution) -- necessary, not sufficient, when stock can block
        all servers (same caveat as every other model in this family)."""
        server_rates = (1.0 / (p.p1 / p.mu1 + (1.0 - p.p1) / p.mu2) for p in self.servers)
        rho = self.l / sum(server_rates)
        if rho >= 1:
            raise ValueError(f"System is unstable: rho={rho} must be < 1 (necessary, not sufficient here)")
        return rho

    def _n_block(self, n: int) -> np.ndarray:
        solver = self._build_solver()
        off = self._n_offsets[n]
        size = self._n_sizes[n]
        m = self.s_max + 1
        return solver.pi0[off * m : off * m + size * m]

    def _mean_in_system(self) -> float:
        """Same shape as EPIC-038's (validated) formula: boundary_term + (c-1)*p_ge_c + mean_extra."""
        solver = self._build_solver()
        m = self.s_max + 1
        c = self.c
        big_m = self._n_rep * m
        boundary_term = sum(n * float(self._n_block(n).sum()) for n in range(c))

        inv = np.linalg.inv(np.eye(big_m) - solver.r)
        ones = np.ones(big_m)
        p_ge_c = float(solver.pi1 @ inv @ ones)
        mean_extra = float(solver.pi1 @ inv @ inv @ ones)
        return boundary_term + (c - 1) * p_ge_c + mean_extra

    def _phase_marginal(self) -> np.ndarray:
        solver = self._build_solver()
        m = self.s_max + 1
        c = self.c
        total = np.zeros(m)
        for n in range(c):
            block = self._n_block(n).reshape(self._n_sizes[n], m)
            total += block.sum(axis=0)
        inv = np.linalg.inv(np.eye(self._n_rep * m) - solver.r)
        vec = (solver.pi1 @ inv).reshape(self._n_rep, m)
        total += vec.sum(axis=0)
        return total

    def _effective_arrival_rate(self) -> float:
        if self.policy == "backorder":
            return self.l
        stockout_prob = float(self._phase_marginal()[0])
        return self.l * (1.0 - stockout_prob)

    def _mean_service_time(self) -> float:
        """E[S] via the same flow-balance shortcut as EPIC-038 -- unaffected by which branch a
        busy server runs, only by occupied count n."""
        solver = self._build_solver()
        m = self.s_max + 1
        c = self.c
        big_m = self._n_rep * m

        boundary_active = 0.0
        for n in range(1, c):
            block = self._n_block(n)
            active = float(block.reshape(self._n_sizes[n], m)[:, 1:].sum())
            boundary_active += n * active

        inv = np.linalg.inv(np.eye(big_m) - solver.r)
        ones = np.ones(big_m)
        zero_stock_mask = np.zeros(big_m)
        zero_stock_mask[0::m] = 1.0
        p_ge_c_active = float(solver.pi1 @ inv @ ones) - float(solver.pi1 @ inv @ zero_stock_mask)

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

    # ------------------------------- waiting-time DISTRIBUTION (EPIC-075)
    def _boundary_stock_probs(self, n: int):
        """Stock distribution at boundary level n < c, summed over configurations."""
        m = self.s_max + 1
        return self._n_block(n).reshape(self._n_sizes[n], m).sum(axis=0)

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
