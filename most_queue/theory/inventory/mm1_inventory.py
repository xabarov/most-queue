"""
M/M/1 queueing-inventory system, (s,S) replenishment policy, backordering or
lost sales.

Customers arrive Poisson(lambda) and are served FCFS at rate mu -- but
service consumes one unit of stock, released only when service completes.
Stock S_max is depleted by completed services; the moment it drops to the
reorder point `s` (0 <= s < S_max; s=0 is the (0,S) special case), an order
for S_max - (current stock) units is placed automatically and arrives after
Exp(theta) (positive lead time), restoring stock to S_max. While stock is 0,
service is blocked (cannot start without a unit in stock); what happens to a
customer arriving during the stockout depends on `policy`:

- "backorder" (default): the customer still queues and waits for
  replenishment (Schwarz M., Daduna H., M/M/1 Queueing systems with
  inventory, Queueing Systems, 2006, doi:10.1007/s11134-006-8710-5; Schwarz
  M., Wichelhaus C., Daduna H., Queueing systems with inventory management
  with random lead times and with backordering, Mathematical Methods of
  Operations Research, 2006, doi:10.1007/s00186-006-0085-1).
- "lost_sales": the customer is turned away instead (Saffari M., Haji R.,
  Hassanzadeh F., The M/M/1 queue with inventory, lost sale, and general
  lead times, Queueing Systems, 2013, doi:10.1007/s11134-012-9337-3).

Exactly a QBD process: level n = number of customers in the system
(unbounded), phase i in {0,...,S_max} = stock level. Solved via the
library's general QBD solver (theory.matrix.qbd.QBDSolver) -- see
docs/roadmaps/queueing_inventory_roadmap.md sec. 2 for the (0,S) backorder
block derivation, docs/roadmaps/queueing_inventory_lost_sales_roadmap.md
sec. 2 for the lost-sales modification (an arrival at stock=0 is not a state
transition at all under lost sales, so A0/B01 lose their phase-0 row and
A1[0,0]/B00[0,0] lose the lambda term), and
docs/roadmaps/queueing_inventory_general_sS_roadmap.md sec. 2 for the general
(s,S) modification: with memoryless (exponential) lead time, "an order is in
transit" is fully determined by i <= s (no extra state bit needed, exactly
as i == 0 fully determined it for (0,S)) -- the replenishment transition is
simply active for every phase i <= s, not just i == 0. s=0 reduces exactly
to the original (0,S) formulas.
"""

from typing import Literal

import numpy as np

from most_queue.structs import QueueingInventoryResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.inventory._wait_phase import (
    build_wait_phase_type,
    phase_type_moments,
    phase_type_tail,
)
from most_queue.theory.matrix.qbd import QBDSolver

Policy = Literal["backorder", "lost_sales"]


class MM1QueueingInventoryCalc(BaseQueue):
    """
    M/M/1 queueing-inventory system with an (s, S) replenishment policy.

    :param s_max: S -- stock level restored to after each replenishment.
    :param s: reorder point, 0 <= s < s_max. s=0 is the (0,S) special case
        (default -- preserves the EPIC-024/026 behaviour exactly).
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
        self.theta = None
        self._solver: QBDSolver | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, mu: float, theta: float):  # pylint: disable=arguments-differ
        """
        :param mu: service rate (consumes one stock unit on completion).
        :param theta: replenishment rate (Exp(theta) lead time), triggered
            automatically whenever stock hits 0.
        """
        if mu <= 0 or theta <= 0:
            raise ValueError(f"mu and theta must be positive, got mu={mu}, theta={theta}")
        self.mu = float(mu)
        self.theta = float(theta)
        self.is_servers_set = True

    # ------------------------------------------------------------ internals
    def _build_solver(self) -> QBDSolver:
        if self._solver is not None:
            return self._solver
        self._check_if_servers_and_sources_set()

        m = self.s_max + 1  # phases: stock level i = 0..S
        lam, mu, theta = self.l, self.mu, self.theta
        lost = self.policy == "lost_sales"

        a0 = lam * np.eye(m)  # arrival: level up, phase unchanged
        if lost:
            a0[0, 0] = 0.0  # an arrival at stock=0 is lost, not a transition

        a2 = np.zeros((m, m))  # service completion: level down, stock -1
        for i in range(1, m):
            a2[i, i - 1] = mu
        # a2[0, :] stays 0: service is blocked when stock is 0

        # Replenishment (rate theta, phase i -> S) is active for every phase
        # i <= s -- an order is "in transit" throughout that range, not just
        # at i == 0 (see module docstring: no extra state bit needed thanks
        # to the memoryless lead time).
        a1 = np.zeros((m, m))
        for i in range(self.s + 1):
            a1[i, self.s_max] = theta
        a1[0, 0] = -theta if lost else -(lam + theta)
        for i in range(1, self.s + 1):
            a1[i, i] = -(lam + mu + theta)  # service still possible (i >= 1) AND an order is in transit
        for i in range(self.s + 1, m):
            a1[i, i] = -(lam + mu)

        b00 = np.zeros((m, m))  # level 0: nothing to serve, no mu-transitions
        for i in range(self.s + 1):
            b00[i, self.s_max] = theta
        b00[0, 0] = -theta if lost else -(lam + theta)
        for i in range(1, self.s + 1):
            b00[i, i] = -(lam + theta)
        for i in range(self.s + 1, m):
            b00[i, i] = -lam

        b01 = a0.copy()
        b10 = a2.copy()

        self._solver = QBDSolver(a0, a1, a2, b00, b01, b10)
        self._solver.solve()
        return self._solver

    def _utilization(self) -> float:
        rho = self.l / self.mu
        if rho >= 1:
            raise ValueError(f"System is unstable: rho={rho} must be < 1 (necessary, not sufficient here)")
        return rho

    def _mean_in_system(self) -> float:
        """E[N] via the matrix-geometric mean formula (level 0 contributes 0)."""
        solver = self._build_solver()
        m = solver.r.shape[0]
        inv = np.linalg.inv(np.eye(m) - solver.r)
        return float(solver.pi1 @ inv @ inv @ np.ones(m))

    def _phase_marginal(self) -> np.ndarray:
        """
        Exact stock-level distribution, summed over all levels n:
        sum_k pi_k = pi0 + pi1 (I - R)^{-1} (matrix-geometric series sum).
        """
        solver = self._build_solver()
        m = solver.r.shape[0]
        inv = np.linalg.inv(np.eye(m) - solver.r)
        return solver.pi0 + solver.pi1 @ inv

    def _effective_arrival_rate(self) -> float:
        """
        Throughput of customers that actually enter the system. Equals lambda
        under backorder (nobody is turned away); under lost_sales it is
        lambda * P(stock > 0), since level n only counts accepted customers
        -- Little's law must use this, not the nominal lambda.
        """
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

    # ------------------------------- waiting-time DISTRIBUTION (EPIC-075)
    def _wait_phase_type(self, level_truncation: int | None = None):
        """
        Phase-type representation ``(alpha, T, p_zero)`` of the waiting time.

        Delegates to the shared builder in
        :mod:`most_queue.theory.inventory._wait_phase`, which carries the full
        literature attribution (the waiting-time distribution for
        queueing-inventory is known theory -- Jeganathan et al. 2021 and
        others; our contribution is the implementation by a tagged-customer
        absorbing chain rather than numerical Laplace inversion).
        """
        self._check_if_servers_and_sources_set()
        solver = self._build_solver()
        n_max = level_truncation or self.calc_params.p_num
        levels = solver.level_probs(n_max + 1)
        return build_wait_phase_type(
            level_phase_probs=lambda n: levels[n],
            c=1,
            s_max=self.s_max,
            s=self.s,
            full_rate=self.mu,
            theta=self.theta,
            n_max=n_max,
            lost_sales=self.policy == "lost_sales",
        )

    def get_w_moments(self, num: int = 4, level_truncation: int | None = None) -> list[float]:
        """
        Exact raw moments of the waiting time W, via the phase-type
        representation (see ``_wait_phase_type`` for the literature context).

        The first moment must agree with :meth:`get_w` -- that is a strong
        cross-check, since ``get_w`` goes through Little's law on the
        stationary distribution while this goes through a tagged-customer
        absorbing chain.
        """
        alpha, t_mat, _ = self._wait_phase_type(level_truncation)
        return phase_type_moments(alpha, t_mat, num)

    def get_tail(self, t: float, level_truncation: int | None = None) -> float:
        """Exact P(W > t) -- deadline-violation probability for the wait."""
        alpha, t_mat, _ = self._wait_phase_type(level_truncation)
        return phase_type_tail(alpha, t_mat, t)

    def get_cdf(self, t: float, level_truncation: int | None = None) -> float:
        """Exact P(W <= t)."""
        return 1.0 - self.get_tail(t, level_truncation)

    def run(self, num_levels: int | None = None) -> QueueingInventoryResults:
        """Solve the QBD and report queue + inventory metrics."""
        start = self._measure_time()
        with self._validate_state():
            utilization = self._utilization()
            p = self.get_p(num_levels)
            v = self.get_v()
            w = self.get_w()
            stock = self.get_stock_distribution()

        # Under lost sales every arrival that finds stock=0 is turned away, so
        # (by PASTA) the loss probability is exactly the stockout probability;
        # under backorder nobody is ever lost.
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
