"""
M/M/c queue with server breakdowns and repairs (Mitrany & Avi-Itzhak,
Management Science, 1968; Neuts & Lucantoni, Management Science, 1979).

Each of the c servers fails independently at rate xi (busy or idle alike) and
is repaired at rate eta; with `repairmen=None` repairs run in parallel
(unlimited crew, so the number of up servers is an independent binomial
birth-death process), otherwise at most R servers are repaired simultaneously.
An interrupted job goes back to the queue (exponential service is memoryless).

Exact solution of the truncated CTMC on states (k jobs, u operational
servers); the truncation level K is grown until the tail mass is negligible.

Stability: lambda < c * mu * eta / (xi + eta) (mean available capacity).
"""

import time

import numpy as np

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.reliability.utils import ctmc_stationary


class MMcBreakdownsCalc(BaseQueue):
    """
    M/M/c with independent server breakdowns and exponential repairs.

    :param n: number of servers c.
    :param repairmen: max simultaneous repairs (None — unlimited).
    """

    def __init__(self, n: int, repairmen: int | None = None):
        super().__init__(n=n)
        self.repairmen = repairmen
        self.l = None
        self.mu = None
        self.xi = None
        self.eta = None
        self.availability = None
        self.up_distribution = None
        self.results = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu: float, xi: float, eta: float):  # pylint: disable=arguments-differ
        """
        :param mu: service rate per (operational) server.
        :param xi: failure rate per server (busy or idle).
        :param eta: repair rate per server under repair.
        """
        self.mu = mu
        self.xi = xi
        self.eta = eta
        self.is_servers_set = True

    def _repair_rate(self, down: int) -> float:
        crew = down if self.repairmen is None else min(down, self.repairmen)
        return crew * self.eta

    def run(self, tail_tol: float = 1e-10, k_start: int = 200, k_max: int = 6400) -> QueueResults:
        """
        Solve the truncated CTMC, growing the truncation until the tail mass
        at the boundary level is below `tail_tol`.
        """
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        avail = self.eta / (self.xi + self.eta)
        capacity = self.n * self.mu * avail
        if self.repairmen is None and self.l >= capacity:
            raise ValueError(f"Unstable: lambda={self.l} >= mean available capacity {capacity:.4f}")

        c = self.n
        k_cap = k_start
        while True:
            n_states = (k_cap + 1) * (c + 1)

            def idx(k, u):
                return k * (c + 1) + u

            trans = []
            for k in range(k_cap + 1):
                for u in range(c + 1):
                    if k < k_cap:
                        trans.append((idx(k, u), idx(k + 1, u), self.l))
                    if k > 0 and u > 0:
                        trans.append((idx(k, u), idx(k - 1, u), min(k, u) * self.mu))
                    if u > 0:
                        trans.append((idx(k, u), idx(k, u - 1), u * self.xi))
                    if u < c:
                        trans.append((idx(k, u), idx(k, u + 1), self._repair_rate(c - u)))
            pi = ctmc_stationary(trans, n_states)
            tail = pi[idx(k_cap, 0) : idx(k_cap, c) + 1].sum()
            if tail < tail_tol or k_cap >= k_max:
                break
            k_cap *= 2

        pi2 = pi.reshape(k_cap + 1, c + 1)
        k_marg = pi2.sum(axis=1)
        u_marg = pi2.sum(axis=0)
        mean_jobs = float(np.dot(np.arange(k_cap + 1), k_marg))

        self.up_distribution = [float(x) for x in u_marg]
        self.availability = float(np.dot(np.arange(c + 1), u_marg)) / c
        v1 = mean_jobs / self.l
        busy = sum(min(k, u) * pi2[k, u] for k in range(k_cap + 1) for u in range(c + 1))

        self.results = QueueResults(
            v=[v1],
            p=[float(x) for x in k_marg],
            utilization=float(busy) / c,
            duration=time.process_time() - start,
        )
        return self.results
