"""
M/Erlang/c queueing-inventory system, c HETEROGENEOUS servers, each with its
OWN Erlang(r_k, rate_k) service-time distribution -- EPIC-041, the CV<=1
complement to EPIC-039's per-server H2 (CV>=1) model.

Per-server state is extended from EPIC-038's binary idle/busy flag to a
(r_k+1)-valued one: idle (-1), or busy at Erlang phase p in {0,...,r_k-1}.
A "config" is a length-c tuple. Unlike H2 (EPIC-039), where a server's
branch never changes once chosen, Erlang's sequential-phase-advance means a
busy server's phase CAN change (p -> p+1) without a departure -- a
same-level ("stay") transition the repeating part of the QBD did not need
in EPIC-039.

Critical correction found via the r_k>1 (non-degenerate) DES cross-check,
NOT the r_k=1 degenerate one (which passes even with the bug, since every
server is trivially "at its last phase" when r_k=1): a config's total
outflow at stock=0 must exclude only the rate of servers AT THEIR LAST
PHASE (whose completion actually consumes a stock unit, hence blocked when
there is no unit) -- NOT the rate of servers at an intermediate phase
(whose advance doesn't touch stock at all, and must never be blocked).
Reusing EPIC-038/039's single-argument diag_block(mu_local) naively lumps
both components together and silently drops outflow from the generator's
diagonal at i=0, producing a non-row-sum-zero matrix and a diverging QBD.
Fixed by splitting outflow into departure_rate (blocked at i=0, feeds
down_block) and advance_rate (never blocked, feeds a new within-level
a1/b00 off-diagonal term) -- see the module's diag_block signature below.

r_k=1 for all k (every busy server trivially "at its last phase") reduces
EXACTLY to MMcQueueingInventoryHeterogeneousCalc (EPIC-038) -- the primary
structural regression, verified to full float64 precision.

See docs/roadmaps/queueing_inventory_heterogeneous_servers_erlang_service_roadmap.md
for the full block derivation and
docs/research/queueing-inventory-heterogeneous-servers-erlang-service-2026.md
for the literature and the bug account.
"""

from itertools import product
from typing import Literal

import numpy as np

from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.structs import QueueingInventoryResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.matrix.qbd import QBDSolver

Policy = Literal["backorder", "lost_sales"]

Config = tuple[int, ...]  # length c, entries in {-1, 0, ..., r_k-1}


def _is_last_phase(config: Config, k: int, servers: list[ErlangParams]) -> bool:
    return config[k] == servers[k].r - 1


def _departure_rate(config: Config, c: int, servers: list[ErlangParams]) -> float:
    return sum(servers[k].mu for k in range(c) if config[k] != -1 and _is_last_phase(config, k, servers))


def _advance_rate(config: Config, c: int, servers: list[ErlangParams]) -> float:
    return sum(servers[k].mu for k in range(c) if config[k] != -1 and not _is_last_phase(config, k, servers))


def _occupied(config: Config) -> int:
    return sum(1 for x in config if x != -1)


def _with(config: Config, k: int, value: int) -> Config:
    updated = list(config)
    updated[k] = value
    return tuple(updated)


class MMcQueueingInventoryHeterogeneousErlangCalc(BaseQueue):
    """
    M/Erlang/c queueing-inventory system: c heterogeneous servers, each with
    its own Erlang(r, rate) service-time distribution, and an (s, S)
    replenishment policy.

    :param c: number of servers (heterogeneous, each with its own Erlang fit).
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
        self.servers: list[ErlangParams] | None = None
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

    def set_servers(self, servers: list[ErlangParams], theta: float):  # pylint: disable=arguments-differ
        """
        :param servers: Erlang service-time parameters of each of the c
            servers, in assignment-priority order (servers[0] preferred
            whenever multiple servers are idle).
        :param theta: replenishment rate (Exp(theta) lead time).
        """
        if len(servers) != self.c:
            raise ValueError(f"expected {self.c} server Erlang params, got {len(servers)}")
        for params in servers:
            if params.r < 1:
                raise ValueError(f"r must be >= 1, got {params.r}")
            if params.mu <= 0:
                raise ValueError(f"per-phase rate must be positive, got {params.mu}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.servers = list(servers)
        self.theta = float(theta)
        self.is_servers_set = True

    def set_servers_from_moments(self, moments_per_server: list[list[float]], theta: float):
        """Fit each server's (r, rate) independently from its own raw moments (fit_erlang)."""
        if len(moments_per_server) != self.c:
            raise ValueError(f"expected {self.c} moment lists, got {len(moments_per_server)}")
        servers = [ErlangDistribution.get_params(moments) for moments in moments_per_server]
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

        def diag_block(dep_rate: float, adv_rate: float) -> np.ndarray:
            """dep_rate: blocked at i=0 (consumes stock on completion).
            adv_rate: NEVER blocked (intermediate phase advance doesn't touch stock)."""
            diag = np.zeros((m, m))
            for i in range(s + 1):
                diag[i, s_max] += theta
            diag[0, 0] = -(adv_rate + theta) if lost else -(lam + adv_rate + theta)
            for i in range(1, s + 1):
                diag[i, i] = -(arrival_row(i) + dep_rate + adv_rate + theta)
            for i in range(s + 1, m):
                diag[i, i] = -(arrival_row(i) + dep_rate + adv_rate)
            return diag

        def down_block(mu_local: float) -> np.ndarray:
            down = np.zeros((m, m))
            for i in range(1, m):
                down[i, i - 1] = mu_local
            return down

        arrival_diag = np.diag([arrival_row(i) for i in range(m)])

        domains = [[-1] + list(range(servers[k].r)) for k in range(c)]
        all_configs = list(product(*domains))
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
        n_rep = len(repeating_configs)
        big_m = n_rep * m

        # ----------------------------------------------- repeating part (n >= c)
        a0 = np.zeros((big_m, big_m))
        a1 = np.zeros((big_m, big_m))
        a2 = np.zeros((big_m, big_m))

        for cfg in repeating_configs:
            ridx = rep_index[cfg]
            lo, hi = ridx * m, (ridx + 1) * m
            a1[lo:hi, lo:hi] = diag_block(_departure_rate(cfg, c, servers), _advance_rate(cfg, c, servers))
            a0[lo:hi, lo:hi] = arrival_diag
            for k in range(c):
                rk = servers[k].mu
                if _is_last_phase(cfg, k, servers):
                    r0 = rep_index[_with(cfg, k, 0)]
                    a2[lo:hi, r0 * m : (r0 + 1) * m] += down_block(rk)
                else:
                    nxt = rep_index[_with(cfg, k, cfg[k] + 1)]
                    a1[lo:hi, nxt * m : (nxt + 1) * m] += rk * np.eye(m)

        # ----------------------------------------------- boundary (n = 0..c-1)
        b00 = np.zeros((m0, m0))
        for n in range(c):
            for cfg in boundary_by_n[n]:
                sidx = subset_index[cfg]
                lo, hi = sidx * m, (sidx + 1) * m
                b00[lo:hi, lo:hi] = diag_block(_departure_rate(cfg, c, servers), _advance_rate(cfg, c, servers))

                if n < c - 1:
                    slot = min(k for k in range(c) if cfg[k] == -1)
                    tidx = subset_index[_with(cfg, slot, 0)]
                    b00[lo:hi, tidx * m : (tidx + 1) * m] += arrival_diag

                for k in range(c):
                    if cfg[k] == -1:
                        continue
                    rk = servers[k].mu
                    if _is_last_phase(cfg, k, servers):
                        tidx = subset_index[_with(cfg, k, -1)]
                        b00[lo:hi, tidx * m : (tidx + 1) * m] += down_block(rk)
                    else:
                        tidx = subset_index[_with(cfg, k, cfg[k] + 1)]
                        b00[lo:hi, tidx * m : (tidx + 1) * m] += rk * np.eye(m)

        b01 = np.zeros((m0, big_m))
        for cfg in boundary_by_n[c - 1]:
            sidx = subset_index[cfg]
            lo, hi = sidx * m, (sidx + 1) * m
            slot = min(k for k in range(c) if cfg[k] == -1)
            ridx = rep_index[_with(cfg, slot, 0)]
            b01[lo:hi, ridx * m : (ridx + 1) * m] += arrival_diag

        b10 = np.zeros((big_m, m0))
        for cfg in repeating_configs:
            ridx = rep_index[cfg]
            lo, hi = ridx * m, (ridx + 1) * m
            for k in range(c):
                if _is_last_phase(cfg, k, servers):
                    rk = servers[k].mu
                    tidx = subset_index[_with(cfg, k, -1)]
                    b10[lo:hi, tidx * m : (tidx + 1) * m] += down_block(rk)

        self._solver = QBDSolver(a0, a1, a2, b00, b01, b10)
        self._solver.solve()
        self._n_offsets = n_offsets
        self._n_sizes = n_sizes
        self._n_rep = n_rep
        return self._solver

    def _utilization(self) -> float:
        """rho = lambda / sum_k(rate_k/r_k), i.e. sum of each server's mean rate (1/E[S_k])."""
        server_rates = (p.mu / p.r for p in self.servers)
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
        """Same shape as EPIC-038/039's (validated) formula: boundary_term + (c-1)*p_ge_c + mean_extra."""
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
        """E[S] via the same flow-balance shortcut as EPIC-038/039 -- unaffected by which phase a
        busy server is in, only by occupied count n."""
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
