"""
M/M/1 retrial queue with two priority classes (Artalejo, Statistica
Neerlandica, 1994; preemptive/non-preemptive retrial priority surveys —
e.g. Operational Research, 2015, doi:10.1007/s12351-015-0175-z).

Class 0 (priority) customers wait in an ordinary FIFO queue when the server
is busy; class 1 (ordinary) customers finding the server busy join the orbit
and retry individually at rate gamma. At a service completion the priority
queue is served first; orbital retrials succeed only when the server is idle
and the priority queue is empty. Service is exponential with class-dependent
rates. Non-preemptive.

Exact solution of the truncated CTMC on states (priority queue q, orbit j,
server state in {idle, serving class 0, serving class 1}).
"""

import time

import numpy as np

from most_queue.structs import MulticlassResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.reliability.utils import ctmc_stationary

IDLE, SERV0, SERV1 = 0, 1, 2


class MM1RetrialPriorityCalc(BaseQueue):
    """
    Two-class M/M/1 retrial queue: priority class queues, ordinary class
    orbits.

    :param gamma: retrial rate of each orbiting (ordinary) customer.
    """

    def __init__(self, gamma: float):
        super().__init__(n=1)
        self.gamma = float(gamma)
        self.l = None
        self.mu = None
        self.mean_priority_queue = None
        self.mean_orbit = None

    def set_sources(self, l: list[float]):  # pylint: disable=arguments-differ
        """:param l: arrival rates [class 0 (priority), class 1 (orbiting)]."""
        if len(l) != 2:
            raise ValueError("Exactly two classes are supported")
        self.l = [float(x) for x in l]
        self.is_sources_set = True

    def set_servers(self, mu: list[float]):  # pylint: disable=arguments-differ
        """:param mu: service rates per class [mu_0, mu_1]."""
        self.mu = [float(x) for x in mu]
        self.is_servers_set = True

    def run(self, tail_tol: float = 1e-10, cap_start: int = 60, cap_max: int = 2000) -> MulticlassResults:
        """
        Solve the truncated CTMC (priority queue and orbit capped, grown
        until the boundary mass is below `tail_tol`).
        """
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        l0, l1 = self.l
        mu0, mu1 = self.mu
        cap = cap_start
        while True:
            m = cap + 1

            def idx(q, j, c):
                return (q * m + j) * 3 + c

            n_states = m * m * 3
            trans = []
            for j in range(m):
                # idle server: priority queue is necessarily empty
                state = idx(0, j, IDLE)
                trans.append((state, idx(0, j, SERV0), l0))
                # an ordinary arrival to an idle server starts service directly
                trans.append((state, idx(0, j, SERV1), l1))
                if j > 0:
                    trans.append((state, idx(0, j - 1, SERV1), j * self.gamma))
            # (q > 0, IDLE) states are unreachable: drain them so the
            # stationary system stays non-singular with zero mass there
            for q in range(1, m):
                for j in range(m):
                    trans.append((idx(q, j, IDLE), idx(0, j, IDLE), 1.0))
            for q in range(m):
                for j in range(m):
                    for c, mu_c in ((SERV0, mu0), (SERV1, mu1)):
                        state = idx(q, j, c)
                        if q < cap:
                            trans.append((state, idx(q + 1, j, c), l0))
                        if j < cap:
                            trans.append((state, idx(q, j + 1, c), l1))
                        if q > 0:
                            trans.append((state, idx(q - 1, j, SERV0), mu_c))
                        else:
                            trans.append((state, idx(0, j, IDLE), mu_c))
            pi = ctmc_stationary(trans, n_states)
            pi3 = pi.reshape(m, m, 3)
            boundary = pi3[cap, :, :].sum() + pi3[:, cap, :].sum()
            if boundary < tail_tol or cap >= cap_max:
                break
            cap *= 2

        qs = np.arange(m)
        q_marg = pi3.sum(axis=(1, 2))
        j_marg = pi3.sum(axis=(0, 2))
        mean_q = float(qs @ q_marg)
        mean_j = float(qs @ j_marg)
        p_serv0 = float(pi3[:, :, SERV0].sum())
        p_serv1 = float(pi3[:, :, SERV1].sum())

        self.mean_priority_queue = mean_q
        self.mean_orbit = mean_j

        # Little: mean time in system per class
        v0 = (mean_q + p_serv0) / l0 if l0 > 0 else 0.0
        v1 = (mean_j + p_serv1) / l1 if l1 > 0 else 0.0
        w = [
            [mean_q / l0 if l0 > 0 else 0.0],
            [mean_j / l1 if l1 > 0 else 0.0],
        ]

        self.results = MulticlassResults(
            w=w,
            v=[[v0], [v1]],
            utilization=p_serv0 + p_serv1,
            duration=time.process_time() - start,
        )
        return self.results
