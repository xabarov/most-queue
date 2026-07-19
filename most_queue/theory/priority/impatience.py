"""
M/M/n + M queue with two priority classes and per-class impatience
(non-preemptive priority Erlang-A).

Two Poisson classes share n exponential servers (class-independent rate mu —
the BCMP-type FCFS condition, which also makes the aggregate process an
ordinary Erlang-A). Class 0 has non-preemptive priority: whenever a server
frees, the longest-waiting class-0 customer is taken first. Waiting customers
of class k abandon at rate theta_k each.

Literature: Choi et al. (Queueing Systems, 2001) — M/M/1 with impatient
high-priority customers; Iravani & Balcioglu (Queueing Systems, 2008) —
priority queues with impatient customers; many-server asymptotics — Atar,
Mandelbaum & Reiman (Ann. Appl. Prob., 2004).

Exact solution of the truncated CTMC on states (busy servers s < n) plus
(q0, q1) queue grid at s = n. Exact special cases used in tests: a single
class reduces to Erlang-A; with theta_0 = theta_1 the TOTAL queue length
distribution coincides with the aggregate Erlang-A (priority only reshuffles
the split between classes).
"""

import time

import numpy as np

from most_queue.structs import MulticlassResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.reliability.utils import ctmc_stationary


class MMnPriorityImpatienceCalc(BaseQueue):
    """
    Two-class non-preemptive priority M/M/n + M (Erlang-A with priorities).

    :param n: number of servers.
    """

    def __init__(self, n: int):
        super().__init__(n=n)
        self.l = None
        self.mu = None
        self.theta = None
        self.mean_queue = None  # per class E[q_k]
        self.abandon_probs = None

    def set_sources(self, l: list[float]):  # pylint: disable=arguments-differ
        """:param l: arrival rates [class 0 (priority), class 1]."""
        if len(l) != 2:
            raise ValueError("Exactly two classes are supported")
        self.l = [float(x) for x in l]
        self.is_sources_set = True

    def set_servers(self, mu: float, theta: list[float]):  # pylint: disable=arguments-differ
        """
        :param mu: service rate (class-independent).
        :param theta: abandonment rate of a waiting customer, per class.
        """
        self.mu = float(mu)
        self.theta = [float(x) for x in theta]
        self.is_servers_set = True

    def run(self, tail_tol: float = 1e-10, q_start: int = 60, q_max: int = 1000) -> MulticlassResults:
        """
        Solve the truncated CTMC (queues capped at q_cap, grown until the
        boundary mass is below `tail_tol`).
        """
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        n = self.n
        l0, l1 = self.l
        lam = l0 + l1
        q_cap = q_start
        while True:
            # states: s = 0..n-1 (empty queues), then grid (q0, q1) at s = n
            m_grid = q_cap + 1

            def idx(q0, q1):
                return n + q0 * m_grid + q1

            n_states = n + m_grid * m_grid
            trans = []
            for s in range(n):
                if s < n - 1:
                    trans.append((s, s + 1, lam))
                else:
                    trans.append((s, idx(0, 0), lam))
                if s > 0:
                    trans.append((s, s - 1, s * self.mu))
            for q0 in range(m_grid):
                for q1 in range(m_grid):
                    state = idx(q0, q1)
                    if q0 < q_cap:
                        trans.append((state, idx(q0 + 1, q1), l0))
                    if q1 < q_cap:
                        trans.append((state, idx(q0, q1 + 1), l1))
                    # service completion: priority class first
                    if q0 > 0:
                        trans.append((state, idx(q0 - 1, q1), n * self.mu))
                    elif q1 > 0:
                        trans.append((state, idx(q0, q1 - 1), n * self.mu))
                    else:
                        trans.append((state, n - 1, n * self.mu))
                    # abandonment
                    if q0 > 0:
                        trans.append((state, idx(q0 - 1, q1), q0 * self.theta[0]))
                    if q1 > 0:
                        trans.append((state, idx(q0, q1 - 1), q1 * self.theta[1]))
            pi = ctmc_stationary(trans, n_states)
            boundary = pi[idx(q_cap, 0) :].sum() + sum(pi[idx(q0, q_cap)] for q0 in range(m_grid))
            if boundary < tail_tol or q_cap >= q_max:
                break
            q_cap *= 2

        grid = pi[n:].reshape(m_grid, m_grid)
        qs = np.arange(m_grid)
        mean_q0 = float(qs @ grid.sum(axis=1))
        mean_q1 = float(qs @ grid.sum(axis=0))

        self.mean_queue = [mean_q0, mean_q1]
        self.abandon_probs = [
            self.theta[0] * mean_q0 / l0 if l0 > 0 else 0.0,
            self.theta[1] * mean_q1 / l1 if l1 > 0 else 0.0,
        ]
        # Little over the queue: mean wait of ALL class-k arrivals
        w = [
            [mean_q0 / l0 if l0 > 0 else 0.0],
            [mean_q1 / l1 if l1 > 0 else 0.0],
        ]
        v = [[w[k][0] + 1.0 / self.mu] for k in range(2)]  # served jobs add one service time

        busy = float(sum(s * pi[s] for s in range(n)) + n * grid.sum())
        self.results = MulticlassResults(
            w=w,
            v=v,
            utilization=busy / n,
            duration=time.process_time() - start,
        )
        return self.results
