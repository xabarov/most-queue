"""
M/M/c queueing-inventory system, c identical servers, (s,S) replenishment
policy, backordering or lost sales.

Direct generalization of MM1QueueingInventoryCalc (theory.inventory.mm1_inventory,
EPIC-024/026/027) to c > 1 servers: each service still consumes one shared
stock unit, released only when service completes. The number of busy
servers is min(n, c) -- a deterministic function of the number of customers
`n`, exactly as in the ordinary M/M/c queue, so no extra state bit is
needed. What changes is that the service rate depends on `n` for
n = 0,...,c-1 (rate n*mu, not all servers saturated yet) and only becomes
level-independent (c*mu) once n >= c -- so there are c distinct boundary
levels instead of one.

QBDSolver supports a single boundary block of arbitrary dimension, so the
fix is to stack the c boundary levels into one super-block of dimension
c*(S+1) (phase = stock level i, nested inside customer-count sub-level
n = 0,...,c-1); the repeating (homogeneous) part starts only once all c
servers are saturated (n >= c), and is identical to the M/M/1 model's
blocks with mu replaced by c*mu. Setting c=1 collapses the super-block to
a single sub-level and reduces exactly to MM1QueueingInventoryCalc -- see
docs/roadmaps/queueing_inventory_multiserver_roadmap.md sec. 2-4 for the
full block derivation and docs/research/queueing-inventory-multiserver-2026.md
for the literature (Yue, Zhao & Yue 2016; Krishnamoorthy, Manikandan &
Dhanya 2015).
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


class MMcQueueingInventoryCalc(BaseQueue):
    """
    M/M/c queueing-inventory system with an (s, S) replenishment policy.

    :param c: number of identical servers, c >= 1.
    :param s_max: S -- stock level restored to after each replenishment.
    :param s: reorder point, 0 <= s < s_max. s=0 is the (0,S) special case.
    :param policy: "backorder" (arrivals wait out a stockout, never lost) or
        "lost_sales" (arrivals during a stockout are turned away).
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        c: int,
        s_max: int,
        s: int = 0,
        policy: Policy = "backorder",
        calc_params: CalcParams | None = None,
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
        self.c = int(c)
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
        :param mu: per-server service rate (consumes one stock unit on completion).
        :param theta: replenishment rate (Exp(theta) lead time), triggered
            automatically whenever stock drops to the reorder point `s`.
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

        c = self.c
        m = self.s_max + 1  # phases: stock level i = 0..S
        m0 = c * m  # boundary super-block: sub-levels n = 0..c-1
        lam, mu, theta = self.l, self.mu, self.theta
        s, s_max = self.s, self.s_max
        lost = self.policy == "lost_sales"

        def arrival_row(i: int) -> float:
            return 0.0 if (lost and i == 0) else lam

        # ----------------------------------------------- repeating part (n >= c)
        a0 = np.diag([arrival_row(i) for i in range(m)])

        a2 = np.zeros((m, m))  # service completion: level down, stock -1, rate c*mu
        for i in range(1, m):
            a2[i, i - 1] = c * mu

        a1 = np.zeros((m, m))
        for i in range(s + 1):
            a1[i, s_max] += theta
        a1[0, 0] = -theta if lost else -(lam + theta)
        for i in range(1, s + 1):
            a1[i, i] = -(arrival_row(i) + c * mu + theta)
        for i in range(s + 1, m):
            a1[i, i] = -(arrival_row(i) + c * mu)

        # ----------------------------------------------- boundary (n = 0..c-1)
        b00 = np.zeros((m0, m0))
        for n in range(c):
            lo, hi = n * m, (n + 1) * m
            diag = np.zeros((m, m))
            for i in range(s + 1):
                diag[i, s_max] += theta
            diag[0, 0] = -theta if lost else -(lam + theta)
            for i in range(1, s + 1):
                diag[i, i] = -(arrival_row(i) + n * mu + theta)
            for i in range(s + 1, m):
                diag[i, i] = -(arrival_row(i) + n * mu)
            b00[lo:hi, lo:hi] = diag

            if n < c - 1:  # arrival n -> n+1, still inside the boundary
                b00[lo:hi, hi : hi + m] = np.diag([arrival_row(i) for i in range(m)])
            if n >= 1:  # service completion n -> n-1, rate n*mu, stock -1
                down = np.zeros((m, m))
                for i in range(1, m):
                    down[i, i - 1] = n * mu
                b00[lo:hi, lo - m : lo] = down

        b01 = np.zeros((m0, m))  # only n=c-1 -> n=c (first repeating level)
        b01[(c - 1) * m : c * m, :] = np.diag([arrival_row(i) for i in range(m)])

        b10 = np.zeros((m, m0))  # only n=c -> n=c-1, rate c*mu (== a2)
        b10[:, (c - 1) * m : c * m] = a2

        self._solver = QBDSolver(a0, a1, a2, b00, b01, b10)
        self._solver.solve()
        return self._solver

    def _utilization(self) -> float:
        rho = self.l / (self.c * self.mu)
        if rho >= 1:
            raise ValueError(f"System is unstable: rho={rho} must be < 1 (necessary, not sufficient here)")
        return rho

    def _pi0_blocks(self) -> np.ndarray:
        """pi0 reshaped into (c, S+1): sub-level n's stock-phase vector as row n."""
        solver = self._build_solver()
        m = self.s_max + 1
        return solver.pi0.reshape(self.c, m)

    def _mean_in_system(self) -> float:
        """
        E[N] = sum_{n=0}^{c-1} n*P(n)  [boundary]
             + (c-1)*P(N>=c) + sum_{j>=1} j*pi1 R^{j-1}  [repeating, shifted by c-1]
        """
        solver = self._build_solver()
        m = self.s_max + 1
        pi0_blocks = self._pi0_blocks()
        boundary_term = sum(n * float(pi0_blocks[n].sum()) for n in range(self.c))

        inv = np.linalg.inv(np.eye(m) - solver.r)
        p_ge_c = float(solver.pi1 @ inv @ np.ones(m))
        mean_extra = float(solver.pi1 @ inv @ inv @ np.ones(m))
        return boundary_term + (self.c - 1) * p_ge_c + mean_extra

    def _phase_marginal(self) -> np.ndarray:
        """Exact stock-level distribution, summed over all levels N."""
        solver = self._build_solver()
        m = self.s_max + 1
        boundary_sum = self._pi0_blocks().sum(axis=0)
        inv = np.linalg.inv(np.eye(m) - solver.r)
        return boundary_sum + solver.pi1 @ inv

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
        pi0_blocks = self._pi0_blocks()

        p = [float(pi0_blocks[k].sum()) for k in range(min(self.c, n_levels))]
        if n_levels > self.c:
            vec = solver.pi1.copy()
            for _ in range(n_levels - self.c):
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
        Mean waiting time, exact -- from the tagged-customer phase-type
        representation (:meth:`get_w_moments`).

        NOTE (bug fixed in EPIC-075). This used to be computed as
        ``E[V] - 1/mu``, which is WRONG for ``c > 1`` whenever stockouts
        actually occur. With several servers running, one of them can consume
        the last stock unit while another customer is still mid-service; that
        customer is then BLOCKED until a replenishment arrives, so its time in
        service is not Exp(mu) and ``E[S] > 1/mu``. Subtracting ``1/mu``
        therefore overstates the wait. Measured example (c=2, S=6, s=0,
        lam=mu=1, theta=20): independent simulation gives E[S] = 1.0058 and
        E[W] = 0.3485 +- 0.0015, the old formula gave 0.3527 (+2.85 sigma),
        the phase-type value gives 0.3472 (-0.93 sigma).

        The single-server model is unaffected -- with one server the stock
        cannot drop while that service is running, so ``E[S] = 1/mu`` exactly
        there.
        """
        self.w = [self.get_w_moments(1)[0]]
        return self.w

    def get_service_time_mean(self) -> float:
        """
        Mean time actually spent in service, ``E[S] = E[V] - E[W]``.

        Exceeds ``1/mu`` when ``c > 1`` and stockouts occur, because a service
        in progress is suspended while stock is out (see :meth:`get_w`).
        """
        v = self.v if self.v is not None else self.get_v()
        w = self.w if self.w is not None else self.get_w()
        return v[0] - w[0]

    def get_stock_distribution(self) -> list[float]:
        """P(stock level = i), i = 0..S, exact (summed over all queue lengths)."""
        return [float(x) for x in self._phase_marginal()]

    # ------------------------------- waiting-time DISTRIBUTION (EPIC-075)
    def _wait_phase_type(self, level_truncation: int | None = None):
        """
        Phase-type representation ``(alpha, T, p_zero)`` of the waiting time.

        Shares the construction with the single-server model -- see
        :mod:`most_queue.theory.inventory._wait_phase` for the full literature
        attribution and the derivation. With ``c`` servers the tagged customer
        needs fewer than ``c`` customers still ahead of it (so that a server is
        free) AND positive stock; the completion rate while it waits is
        ``min(r, c) * mu``, matching the ``min(n, c)`` busy-server count this
        model is built on.
        """
        self._check_if_servers_and_sources_set()
        solver = self._build_solver()
        n_max = level_truncation or self.calc_params.p_num
        pi0_blocks = self._pi0_blocks()

        repeating: list[np.ndarray] = []
        if n_max >= self.c:
            vec = solver.pi1.copy()
            for _ in range(n_max - self.c + 1):
                repeating.append(vec)
                vec = vec @ solver.r

        def level_phase_probs(n: int) -> np.ndarray:
            if n < self.c:
                return pi0_blocks[n]
            return repeating[n - self.c]

        return build_wait_phase_type(
            level_phase_probs=level_phase_probs,
            c=self.c,
            s_max=self.s_max,
            s=self.s,
            full_rate=self.c * self.mu,
            theta=self.theta,
            n_max=n_max,
            lost_sales=self.policy == "lost_sales",
        )

    def get_w_moments(self, num: int = 4, level_truncation: int | None = None) -> list[float]:
        """
        Exact raw moments of the waiting time W.

        The first moment must agree with :meth:`get_w`, which reaches the same
        quantity through an entirely independent path (Little's law on the
        matrix-geometric stationary distribution) -- a strong cross-check.
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
