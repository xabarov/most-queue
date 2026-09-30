"""
M/M/1 queueing-inventory system with Erlang(r, rate)-fitted replenishment
lead time (not Exp(theta)) -- EPIC-040, the supply-side complement to
EPIC-035/036/039's demand-side (service) phase-type augmentations.

Unlike every other phase-type augmentation in this library (which adds a
phase dimension to EVERY stock/service state), phase-type replenishment
only needs a phase dimension where an order can be in transit -- stock
levels i <= s. Levels i > s have no pending order and need no phase at
all. The augmented state splits into a PLAIN zone (i in {s+1,...,S}, no
phase) and a PENDING zone (i in {0,...,s}, each with r Erlang sub-phases).
A service completion crossing from the plain zone's lowest level (i=s+1)
into the pending zone (i=s) is exactly the moment an order is placed
(phase initialized to 0); every other service completion within the
pending zone preserves the current phase (the order's own progress is
unaffected by further stock depletion); the only route back to the plain
zone is a replenishment completion (last phase finishes), which jumps
stock directly to S regardless of how low it drifted meanwhile. None of
this touches the arrival/service LEVEL structure of the QBD (number of
customers n) -- the augmentation is entirely local to each level's phase
block (diag_block/down_block/arrival_diag), unlike EPIC-039 where the
repeating part itself needed a bigger phase space.

r=1 (Erlang collapses to Exponential) reduces EXACTLY to
MM1QueueingInventoryCalc with theta=rate -- the primary regression,
verified to full float64 precision.

A validation trap avoided before shipping: the first numeric prototype
held the Erlang `rate` fixed while varying `r`, which (correctly, by
mean = r/rate) changes the MEAN lead time, not just its CV -- producing an
apparently unstable system (QBD log-reduction overflow, non-decaying tail
mass in a truncated cross-check) that looked like a construction bug but
was actually a correctly-modeled, much-slower-replenishment regime.
Re-testing with rate = r / mean_lead (holding the mean fixed across r)
resolved it immediately -- the same "hold the mean constant when varying
phase count" lesson as EPIC-035's k vs k*rate confusion.

Mean-only (E[V], E[W] via Little's law), same scope as every other model
in this family. Reserve: H2 replenishment (CV>=1); porting this technique
to MMcQueueingInventoryCalc and the heterogeneous-server calculators
(EPIC-038/039) -- the augmentation doesn't interact with the server-busy
dimension at all, so should port directly, not attempted here.

See docs/roadmaps/queueing_inventory_phase_type_replenishment_roadmap.md
for the full block derivation and
docs/research/queueing-inventory-phase-type-replenishment-2026.md for the
literature.
"""

from typing import Literal

import numpy as np

from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.structs import QueueingInventoryResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.matrix.qbd import QBDSolver

Policy = Literal["backorder", "lost_sales"]


class MM1QueueingInventoryErlangReplenishmentCalc(BaseQueue):
    """
    M/M/1 queueing-inventory system with an (s, S) replenishment policy and
    Erlang(r, rate)-fitted (not exponential) lead time.

    :param s_max: S -- stock level restored to after each replenishment.
    :param s: reorder point, 0 <= s < s_max. s=0 is the (0,S) special case.
    :param policy: "backorder" (arrivals wait out a stockout, never lost) or
        "lost_sales" (arrivals during a stockout are turned away).
    """

    def __init__(self, s_max: int, s: int = 0, policy: Policy = "backorder", calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)
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
        self.mu = None
        self.r = None
        self.rate = None
        self._solver: QBDSolver | None = None
        self._m_plain: int | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, mu: float, r: int, rate: float):  # pylint: disable=arguments-differ
        """
        :param mu: service rate (consumes one stock unit on completion).
        :param r: number of Erlang phases of the replenishment lead time.
        :param rate: per-phase Erlang rate; mean lead time = r / rate.
        """
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.mu = float(mu)
        self.r = int(r)
        self.rate = float(rate)
        self.is_servers_set = True

    def set_replenishment_from_moments(self, mu: float, moments: list[float]):
        """Fit (r, rate) of the replenishment lead time from raw moments (mean, E[L^2], ...)."""
        params: ErlangParams = ErlangDistribution.get_params(moments)
        self.set_servers(mu=mu, r=params.r, rate=params.mu)

    # ------------------------------------------------------------ internals
    def _indices(self):
        s, s_max, r = self.s, self.s_max, self.r
        m_plain = s_max - s

        def plain_idx(i: int) -> int:
            return i - (s + 1)

        def pending_idx(i: int, p: int) -> int:
            return m_plain + i * r + p

        return m_plain, plain_idx, pending_idx

    def _build_solver(self) -> QBDSolver:
        if self._solver is not None:
            return self._solver
        self._check_if_servers_and_sources_set()

        lam, mu, r, rate = self.l, self.mu, self.r, self.rate
        s, s_max = self.s, self.s_max
        lost = self.policy == "lost_sales"
        m_plain, plain_idx, pending_idx = self._indices()
        m_pending = (s + 1) * r
        big_m = m_plain + m_pending

        def arrival_row(i: int) -> float:
            return 0.0 if (lost and i == 0) else lam

        a0 = np.zeros((big_m, big_m))
        for i in range(s + 1, s_max + 1):
            a0[plain_idx(i), plain_idx(i)] = arrival_row(i)
        for i in range(s + 1):
            for p in range(r):
                a0[pending_idx(i, p), pending_idx(i, p)] = arrival_row(i)

        a2 = np.zeros((big_m, big_m))
        for i in range(s + 2, s_max + 1):
            a2[plain_idx(i), plain_idx(i - 1)] = mu
        a2[plain_idx(s + 1), pending_idx(s, 0)] = mu
        for i in range(1, s + 1):
            for p in range(r):
                a2[pending_idx(i, p), pending_idx(i - 1, p)] = mu

        def replenishment_transitions(block: np.ndarray, include_mu: bool):
            for i in range(s + 1, s_max + 1):
                out = arrival_row(i) + (mu if include_mu else 0.0)
                block[plain_idx(i), plain_idx(i)] = -out
            for i in range(s + 1):
                for p in range(r):
                    out = arrival_row(i) + (mu if (include_mu and i >= 1) else 0.0) + rate
                    block[pending_idx(i, p), pending_idx(i, p)] = -out
                    if p < r - 1:
                        block[pending_idx(i, p), pending_idx(i, p + 1)] = rate
                    else:
                        block[pending_idx(i, p), plain_idx(s_max)] = rate

        a1 = np.zeros((big_m, big_m))
        replenishment_transitions(a1, include_mu=True)

        b00 = np.zeros((big_m, big_m))
        replenishment_transitions(b00, include_mu=False)

        b01 = a0.copy()
        b10 = a2.copy()

        self._solver = QBDSolver(a0, a1, a2, b00, b01, b10)
        self._solver.solve()
        self._m_plain = m_plain
        return self._solver

    def _utilization(self) -> float:
        rho = self.l / self.mu
        if rho >= 1:
            raise ValueError(f"System is unstable: rho={rho} must be < 1 (necessary, not sufficient here)")
        return rho

    def _mean_in_system(self) -> float:
        solver = self._build_solver()
        big_m = solver.r.shape[0]
        inv = np.linalg.inv(np.eye(big_m) - solver.r)
        return float(solver.pi1 @ inv @ inv @ np.ones(big_m))

    def _phase_marginal(self) -> np.ndarray:
        """P(stock level = i), i = 0..S -- collapses the Erlang sub-phases of the pending zone."""
        solver = self._build_solver()
        big_m = solver.r.shape[0]
        inv = np.linalg.inv(np.eye(big_m) - solver.r)
        marginal = solver.pi0 + solver.pi1 @ inv

        m_plain, plain_idx, pending_idx = self._indices()
        result = np.zeros(self.s_max + 1)
        for i in range(self.s + 1, self.s_max + 1):
            result[i] = marginal[plain_idx(i)]
        for i in range(self.s + 1):
            result[i] = sum(marginal[pending_idx(i, p)] for p in range(self.r))
        return result

    def _effective_arrival_rate(self) -> float:
        if self.policy == "backorder":
            return self.l
        stockout_prob = float(self._phase_marginal()[0])
        return self.l * (1.0 - stockout_prob)

    # ------------------------------------------------------------- results
    def get_p(self, num_levels: int | None = None) -> list[float]:
        """Probabilities of the number of customers in the system (levels)."""
        self._check_if_servers_and_sources_set()
        solver = self._build_solver()
        n = num_levels or self.calc_params.p_num
        self.p = solver.marginal_level_probs(n)
        return self.p

    def get_v(self) -> list[float]:
        """Mean sojourn time (first moment only): E[V] = E[N] / lambda_effective (Little's law)."""
        self.v = [self._mean_in_system() / self._effective_arrival_rate()]
        return self.v

    def get_w(self) -> list[float]:
        """Mean waiting time: E[W] = E[V] - E[service] (V = W + S for FCFS, always exact)."""
        v = self.v if self.v is not None else self.get_v()
        self.w = [v[0] - 1.0 / self.mu]
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
