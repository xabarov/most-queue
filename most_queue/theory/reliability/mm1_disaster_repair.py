"""
M/M/1 queue with disasters and a repair phase (Towsley & Tripathi, Operations
Research Letters, 1991: server failures with queue flushing).

A disaster (rate delta) wipes out all jobs AND sends the server to repair for
an exponential time (rate eta); arrivals during the repair join the queue and
wait. This strengthens the library's negative-arrivals stack, where a DISASTER
clears the system but the server is instantly operational: eta -> infinity
recovers that model.

Exact solution of the truncated CTMC on states (k jobs, phase in {up, down}).
The system is always stable (disasters flush the queue), but the truncation
grows until the tail is negligible.
"""

import time

import numpy as np

from most_queue.structs import NegativeArrivalsResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.reliability.utils import ctmc_stationary

UP, DOWN = 0, 1


class MM1DisasterRepairCalc(BaseQueue):
    """
    M/M/1 with disasters (queue flush) followed by an exponential repair.
    """

    def __init__(self):
        super().__init__(n=1)
        self.l = None
        self.mu = None
        self.delta = None
        self.eta = None
        self.down_prob = None
        self.results = None

    def set_sources(self, l: float, delta: float):  # pylint: disable=arguments-differ
        """
        :param l: arrival rate.
        :param delta: disaster rate (active while the server is up).
        """
        self.l = l
        self.delta = delta
        self.is_sources_set = True

    def set_servers(self, mu: float, eta: float):  # pylint: disable=arguments-differ
        """
        :param mu: service rate.
        :param eta: repair rate after a disaster.
        """
        self.mu = mu
        self.eta = eta
        self.is_servers_set = True

    def run(self, tail_tol: float = 1e-10, k_start: int = 400, k_max: int = 25600) -> NegativeArrivalsResults:
        """
        Solve the truncated two-phase CTMC.
        """
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        k_cap = k_start
        while True:

            def idx(k, ph):
                return 2 * k + ph

            trans = []
            for k in range(k_cap + 1):
                if k < k_cap:
                    trans.append((idx(k, UP), idx(k + 1, UP), self.l))
                    trans.append((idx(k, DOWN), idx(k + 1, DOWN), self.l))
                if k > 0:
                    trans.append((idx(k, UP), idx(k - 1, UP), self.mu))
                # disaster: flush all jobs, server goes down
                trans.append((idx(k, UP), idx(0, DOWN), self.delta))
                trans.append((idx(k, DOWN), idx(k, UP), self.eta))
            pi = ctmc_stationary(trans, 2 * (k_cap + 1))
            if pi[2 * k_cap :].sum() < tail_tol or k_cap >= k_max:
                break
            k_cap *= 2

        pi2 = pi.reshape(k_cap + 1, 2)
        k_marg = pi2.sum(axis=1)
        mean_jobs = float(np.dot(np.arange(k_cap + 1), k_marg))

        self.down_prob = float(pi2[:, DOWN].sum())
        served_rate = self.mu * float(pi2[1:, UP].sum())
        q = served_rate / self.l  # probability that an arriving job is eventually served

        self.results = NegativeArrivalsResults(
            v=[mean_jobs / self.l],
            p=[float(x) for x in k_marg],
            utilization=float(pi2[1:, UP].sum()),
            q=float(q),
            duration=time.process_time() - start,
        )
        return self.results
