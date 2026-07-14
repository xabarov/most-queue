"""
M/M/1 retrial queue with an unreliable server (Wang, Cao & Li, Queueing
Systems, 2001; Artalejo, Statistica Neerlandica, 1994).

An arriving job that finds the server free starts service; otherwise it joins
the orbit and retries at rate gamma per orbiting job. The server fails during
service at rate xi (active breakdowns); the interrupted job goes back to the
orbit and the server is repaired at rate eta. Arrivals during a repair also
join the orbit; retrials are unsuccessful while the server is down.

Exact solution of the orbit-truncated CTMC on states
(j jobs in orbit, server phase in {idle, busy, down}).
"""

import time

import numpy as np

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.reliability.utils import ctmc_stationary

IDLE, BUSY, DOWN = 0, 1, 2


class MM1RetrialUnreliableCalc(BaseQueue):
    """
    M/M/1 retrial queue with active server breakdowns and repairs.

    :param gamma: retrial rate of each orbiting job.
    """

    def __init__(self, gamma: float):
        super().__init__(n=1)
        self.gamma = gamma
        self.l = None
        self.mu = None
        self.xi = None
        self.eta = None
        self.availability = None
        self.mean_orbit = None
        self.results = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu: float, xi: float = 0.0, eta: float = 1.0):  # pylint: disable=arguments-differ
        """
        :param mu: service rate.
        :param xi: failure rate during service (active breakdowns).
        :param eta: repair rate.
        """
        self.mu = mu
        self.xi = xi
        self.eta = eta
        self.is_servers_set = True

    def run(self, tail_tol: float = 1e-10, j_start: int = 200, j_max: int = 12800) -> QueueResults:
        """
        Solve the orbit-truncated CTMC.
        """
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        j_cap = j_start
        while True:

            def idx(j, s):
                return 3 * j + s

            trans = []
            for j in range(j_cap + 1):
                # idle server
                trans.append((idx(j, IDLE), idx(j, BUSY), self.l))
                if j > 0:
                    trans.append((idx(j, IDLE), idx(j - 1, BUSY), j * self.gamma))
                # busy server
                if j < j_cap:
                    trans.append((idx(j, BUSY), idx(j + 1, BUSY), self.l))
                trans.append((idx(j, BUSY), idx(j, IDLE), self.mu))
                if self.xi > 0 and j < j_cap:
                    trans.append((idx(j, BUSY), idx(j + 1, DOWN), self.xi))
                # server down (repair); retrials are blocked
                if j < j_cap:
                    trans.append((idx(j, DOWN), idx(j + 1, DOWN), self.l))
                trans.append((idx(j, DOWN), idx(j, IDLE), self.eta))
            pi = ctmc_stationary(trans, 3 * (j_cap + 1))
            if pi[3 * j_cap :].sum() < tail_tol or j_cap >= j_max:
                break
            j_cap *= 2

        pi2 = pi.reshape(j_cap + 1, 3)
        j_marg = pi2.sum(axis=1)
        mean_orbit = float(np.dot(np.arange(j_cap + 1), j_marg))
        p_busy = float(pi2[:, BUSY].sum())
        p_down = float(pi2[:, DOWN].sum())

        self.mean_orbit = mean_orbit
        self.availability = 1.0 - p_down
        mean_in_system = mean_orbit + p_busy

        self.results = QueueResults(
            v=[mean_in_system / self.l],
            p=[float(x) for x in j_marg],
            utilization=p_busy,
            duration=time.process_time() - start,
        )
        return self.results
