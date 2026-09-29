"""
M/M/1 queueing-inventory system, (0,S) replenishment policy, backordering.

Customers arrive Poisson(lambda) and are served FCFS at rate mu -- but
service consumes one unit of stock, released only when service completes.
Stock S_max is depleted by completed services; the moment it hits 0, an
order for S_max units is placed automatically ((0,S) policy) and arrives
after Exp(theta) (positive lead time). While stock is 0, arriving customers
still queue (backordering, not lost sales) but the server is blocked --
service cannot start without a unit in stock.

Classic formulation: Schwarz M., Daduna H., M/M/1 Queueing systems with
inventory, Queueing Systems, 2006, doi:10.1007/s11134-006-8710-5; Schwarz M.,
Wichelhaus C., Daduna H., Queueing systems with inventory management with
random lead times and with backordering, Mathematical Methods of Operations
Research, 2006, doi:10.1007/s00186-006-0085-1.

Exactly a QBD process: level n = number of customers in the system
(unbounded), phase i in {0,...,S_max} = stock level. Solved via the
library's general QBD solver (theory.matrix.qbd.QBDSolver) -- see
docs/roadmaps/queueing_inventory_roadmap.md sec. 2 for the block derivation.
"""

import numpy as np

from most_queue.structs import QueueingInventoryResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.matrix.qbd import QBDSolver


class MM1QueueingInventoryCalc(BaseQueue):
    """
    M/M/1 queueing-inventory system with a (0, S) replenishment policy and
    backordering (arriving customers wait, never lost, even during a
    stockout -- only service is blocked).

    :param s_max: S -- stock level restored to after each replenishment.
    """

    def __init__(self, s_max: int, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)
        if s_max < 1:
            raise ValueError(f"s_max must be >= 1, got {s_max}")
        self.s_max = int(s_max)
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

        a0 = lam * np.eye(m)  # arrival: level up, phase unchanged

        a2 = np.zeros((m, m))  # service completion: level down, stock -1
        for i in range(1, m):
            a2[i, i - 1] = mu
        # a2[0, :] stays 0: service is blocked when stock is 0

        a1 = np.zeros((m, m))
        a1[0, self.s_max] = theta  # replenishment: same level, stock 0 -> S
        a1[0, 0] = -(lam + theta)
        for i in range(1, m):
            a1[i, i] = -(lam + mu)

        b00 = np.zeros((m, m))  # level 0: nothing to serve, no mu-transitions
        b00[0, self.s_max] = theta
        b00[0, 0] = -(lam + theta)
        for i in range(1, m):
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

    # ------------------------------------------------------------- results
    def get_p(self, num_levels: int | None = None) -> list[float]:
        """Probabilities of the number of customers in the system (levels)."""
        self._check_if_servers_and_sources_set()
        solver = self._build_solver()
        n = num_levels or self.calc_params.p_num
        self.p = solver.marginal_level_probs(n)
        return self.p

    def get_v(self) -> list[float]:
        """Mean sojourn time (first moment only): E[V] = E[N] / lambda (Little's law)."""
        self.v = [self._mean_in_system() / self.l]
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

        result = QueueingInventoryResults(
            v=v,
            w=w,
            p=p,
            utilization=utilization,
            stock_distribution=stock,
            stockout_prob=stock[0],
            fill_rate=1.0 - stock[0],
        )
        self._set_duration(result, start)
        return result
