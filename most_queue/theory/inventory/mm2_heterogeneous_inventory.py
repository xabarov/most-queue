"""
M/M/2 queueing-inventory system, two HETEROGENEOUS servers (mu1 != mu2),
(s,S) replenishment policy, backordering or lost sales.

Combines two already-validated techniques from this library rather than
introducing new math from scratch: the state-splitting technique for two
heterogeneous exponential servers (Krishnamoorthi B., On Poisson Queue with
Two Heterogeneous Servers, Operations Research, 1963, doi:10.1287/opre.11.3.321
-- already reused in EPIC-023 machine-repair and EPIC-025 priority models),
and the stacked-boundary-superblock QBD trick from MMcQueueingInventoryCalc
(theory.inventory.mmc_inventory, EPIC-028).

State: (n, config, i) -- n customers in system, stock level i, and config in
{1, 2} disambiguating *which* server is the sole busy one, needed only when
n=1 (n=0: nobody busy; n>=2: both servers busy, unambiguous). Same blocking
convention as the whole queueing-inventory family: service consumes one
stock unit at completion (not reserved at start), fully blocked at i=0.

Phase space by level: n=0 has m=S+1 states (stock only); n=1 has 2m states
(config x stock) -- these two levels are stacked into one m0=3m boundary
super-block for QBDSolver, exactly like EPIC-028's c identical boundary
sub-levels, just with a qualitatively different (not just narrower) n=1
level. n>=2 is the homogeneous repeating part (m states, mu1+mu2), same
shape as EPIC-028's c=2 case.

The one genuinely new piece beyond a mechanical combination: the departure
transition from the first repeating level (n=2) down into the boundary
(n=1) must *split* by which server just finished -- rate mu1 leaves server
2 as the sole survivor (config=2), rate mu2 leaves server 1 (config=1) --
the same "disambiguate on departure" step EPIC-023/025 use for a flat CTMC,
here feeding a QBD B10 block instead. Setting mu1=mu2 must reduce exactly to
MMcQueueingInventoryCalc(c=2, ...) -- the primary regression test. See
docs/roadmaps/queueing_inventory_heterogeneous_servers_roadmap.md for the
full block derivation and docs/research/queueing-inventory-heterogeneous-servers-2026.md
for the literature.
"""

from typing import Literal

import numpy as np

from most_queue.structs import QueueingInventoryResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.matrix.qbd import QBDSolver

Policy = Literal["backorder", "lost_sales"]


class MM2QueueingInventoryHeterogeneousCalc(BaseQueue):
    """
    M/M/2 queueing-inventory system with two heterogeneous servers and an
    (s, S) replenishment policy.

    :param s_max: S -- stock level restored to after each replenishment.
    :param s: reorder point, 0 <= s < s_max. s=0 is the (0,S) special case.
    :param policy: "backorder" (arrivals wait out a stockout, never lost) or
        "lost_sales" (arrivals during a stockout are turned away).
    """

    def __init__(self, s_max: int, s: int = 0, policy: Policy = "backorder", calc_params: CalcParams | None = None):
        super().__init__(n=2, calc_params=calc_params)
        if s_max < 1:
            raise ValueError(f"s_max must be >= 1, got {s_max}")
        if not 0 <= s < s_max:
            raise ValueError(f"reorder point s must satisfy 0 <= s < s_max, got s={s}, s_max={s_max}")
        if policy not in ("backorder", "lost_sales"):
            raise ValueError(f"policy must be 'backorder' or 'lost_sales', got {policy!r}")
        self.s_max = int(s_max)
        self.s = int(s)
        self.policy: Policy = policy
        self.l = None
        self.mu1 = None
        self.mu2 = None
        self.theta = None
        self._solver: QBDSolver | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, mu1: float, mu2: float, theta: float):  # pylint: disable=arguments-differ
        """
        :param mu1: service rate of server 1 (preferred when both are idle).
        :param mu2: service rate of server 2.
        :param theta: replenishment rate (Exp(theta) lead time), triggered
            automatically whenever stock drops to the reorder point `s`.
        """
        if mu1 <= 0 or mu2 <= 0 or theta <= 0:
            raise ValueError(f"mu1, mu2 and theta must be positive, got mu1={mu1}, mu2={mu2}, theta={theta}")
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.theta = float(theta)
        self.is_servers_set = True

    # ------------------------------------------------------------ internals
    def _build_solver(self) -> QBDSolver:
        if self._solver is not None:
            return self._solver
        self._check_if_servers_and_sources_set()

        m = self.s_max + 1  # phases: stock level i = 0..S
        m0 = 3 * m  # boundary super-block: n=0 (m) + n=1,config1 (m) + n=1,config2 (m)
        lam, mu1, mu2, theta = self.l, self.mu1, self.mu2, self.theta
        s, s_max = self.s, self.s_max
        lost = self.policy == "lost_sales"
        mu_sum = mu1 + mu2

        def arrival_row(i: int) -> float:
            return 0.0 if (lost and i == 0) else lam

        def diag_block(mu_local: float) -> np.ndarray:
            """Local diagonal block for a level with a single active service rate mu_local
            (mu_local=0 for n=0, where nobody is being served)."""
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
            """Departure: i -> i-1 (i>=1), rate mu_local."""
            down = np.zeros((m, m))
            for i in range(1, m):
                down[i, i - 1] = mu_local
            return down

        arrival_diag = np.diag([arrival_row(i) for i in range(m)])

        # ----------------------------------------------- repeating part (n >= 2)
        a0 = arrival_diag.copy()
        a2 = down_block(mu_sum)
        a1 = diag_block(mu_sum)

        # ----------------------------------------------- boundary (n = 0, 1)
        b00 = np.zeros((m0, m0))
        b00[0:m, 0:m] = diag_block(0.0)  # n=0: nobody serving
        b00[m : 2 * m, m : 2 * m] = diag_block(mu1)  # n=1, config=1 (server 1 busy)
        b00[2 * m : 3 * m, 2 * m : 3 * m] = diag_block(mu2)  # n=1, config=2 (server 2 busy)

        b00[0:m, m : 2 * m] = arrival_diag  # n=0 -> n=1: assign to server 1 (tie-break convention)
        b00[m : 2 * m, 0:m] = down_block(mu1)  # n=1,config=1 -> n=0: server 1 finishes
        b00[2 * m : 3 * m, 0:m] = down_block(mu2)  # n=1,config=2 -> n=0: server 2 finishes

        b01 = np.zeros((m0, m))  # n=1 -> n=2 (first repeating level): second server grabbed
        b01[m : 2 * m, :] = arrival_diag
        b01[2 * m : 3 * m, :] = arrival_diag

        b10 = np.zeros((m, m0))  # n=2 -> n=1: split by which server just finished
        b10[:, m : 2 * m] = down_block(mu2)  # server 2 finished -> server 1 is the survivor (config=1)
        b10[:, 2 * m : 3 * m] = down_block(mu1)  # server 1 finished -> server 2 is the survivor (config=2)

        self._solver = QBDSolver(a0, a1, a2, b00, b01, b10)
        self._solver.solve()
        return self._solver

    def _utilization(self) -> float:
        rho = self.l / (self.mu1 + self.mu2)
        if rho >= 1:
            raise ValueError(f"System is unstable: rho={rho} must be < 1 (necessary, not sufficient here)")
        return rho

    def _pi0_level_blocks(self) -> tuple[np.ndarray, np.ndarray]:
        """pi0 split into (n=0 block, n=1 block) -- the latter still config x stock (2m)."""
        solver = self._build_solver()
        m = self.s_max + 1
        pi0 = solver.pi0
        return pi0[:m], pi0[m : 3 * m]

    def _mean_in_system(self) -> float:
        """Same level-counting shape as MMcQueueingInventoryCalc(c=2): boundary spans
        levels n=0,1, repeating starts at n=2 (QBD level j -> N=1+j)."""
        solver = self._build_solver()
        m = self.s_max + 1
        _, n1_block = self._pi0_level_blocks()
        boundary_term = float(n1_block.sum())  # 0*P(0) + 1*P(1)

        inv = np.linalg.inv(np.eye(m) - solver.r)
        p_ge_2 = float(solver.pi1 @ inv @ np.ones(m))
        mean_extra = float(solver.pi1 @ inv @ inv @ np.ones(m))
        return boundary_term + p_ge_2 + mean_extra

    def _phase_marginal(self) -> np.ndarray:
        """Exact stock-level distribution, summed over all levels N."""
        solver = self._build_solver()
        m = self.s_max + 1
        n0_block, n1_block = self._pi0_level_blocks()
        n1_collapsed = n1_block[:m] + n1_block[m:]  # sum over config
        inv = np.linalg.inv(np.eye(m) - solver.r)
        return n0_block + n1_collapsed + solver.pi1 @ inv

    def _effective_arrival_rate(self) -> float:
        """lambda under backorder; lambda*(1 - stockout_prob) under lost_sales."""
        if self.policy == "backorder":
            return self.l
        stockout_prob = float(self._phase_marginal()[0])
        return self.l * (1.0 - stockout_prob)

    # ------------------------------------------------------------- results
    def get_p(self, num_levels: int | None = None) -> list[float]:
        """Probabilities of the number of customers in the system (levels)."""
        self._check_if_servers_and_sources_set()
        solver = self._build_solver()
        n_levels = num_levels or self.calc_params.p_num
        n0_block, n1_block = self._pi0_level_blocks()

        p = [float(n0_block.sum()), float(n1_block.sum())][: min(2, n_levels)]
        if n_levels > 2:
            vec = solver.pi1.copy()
            for _ in range(n_levels - 2):
                p.append(float(vec.sum()))
                vec = vec @ solver.r
        self.p = p
        return self.p

    def get_v(self) -> list[float]:
        """Mean sojourn time (first moment only): E[V] = E[N] / lambda_effective (Little's law)."""
        self.v = [self._mean_in_system() / self._effective_arrival_rate()]
        return self.v

    def _mean_service_time(self) -> float:
        """
        E[S], the mean service time actually experienced by a customer -- a
        weighted average of 1/mu1 and 1/mu2, weighted by the fraction of
        customers each server ends up serving. By flow balance in steady
        state, throughput via server k is P(server k actively serving)*mu_k,
        and every accepted customer departs via exactly one server, so
        fraction_k = P(busy_k)*mu_k / lambda_eff. Substituting into
        E[S] = sum_k fraction_k * (1/mu_k) collapses the mu_k's:

            E[S] = (P(busy1) + P(busy2)) / lambda_eff

        "Actively serving" requires i >= 1 (service is blocked at i=0, same
        convention as the rest of this family) -- reduces exactly to 1/mu at
        mu1=mu2=mu (the MMcQueueingInventoryCalc(c=2) regression case).
        """
        solver = self._build_solver()
        m = self.s_max + 1
        _, n1_block = self._pi0_level_blocks()
        n1_config1, n1_config2 = n1_block[:m], n1_block[m:]

        inv = np.linalg.inv(np.eye(m) - solver.r)
        p_ge_2_active = float(solver.pi1 @ inv @ np.ones(m)) - float(solver.pi1 @ inv[:, 0])
        p_busy1 = float(n1_config1[1:].sum()) + p_ge_2_active
        p_busy2 = float(n1_config2[1:].sum()) + p_ge_2_active

        return (p_busy1 + p_busy2) / self._effective_arrival_rate()

    def get_w(self) -> list[float]:
        """Mean waiting time: E[W] = E[V] - E[service] (V = W + S for FCFS, always exact)."""
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
