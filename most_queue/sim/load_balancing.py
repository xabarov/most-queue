"""
Discrete-event simulator for load-balancing / dispatching over N parallel
single-server (M/M/1) queues.

A central dispatcher routes each Poisson arrival to a server according to a
policy: random, power-of-d (sample d servers, join the shortest), JSQ (shortest
of all), or JIQ (an idle server if any, else random). Validates the mean-field
`LoadBalancingMeanField` calculator at finite N.
"""

import numpy as np

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import QueueResults


class LoadBalancingSim(BaseSimulationCore):
    """
    :param n_servers: number of servers in the pool.
    :param policy: "random", "power-of-d", "jsq", or "jiq".
    :param d: sampled servers for power-of-d.
    :param seed: RNG seed.
    """

    def __init__(self, n_servers: int, policy: str = "power-of-d", d: int = 2, seed: int | None = None):
        super().__init__(seed=seed)
        self.N = n_servers
        self.policy = policy.lower()
        self.d = d
        self.rho = None
        self.mu = None
        self.is_sources_set = False
        self.is_servers_set = False

    def set_sources(self, rho: float):
        """:param rho: per-server offered load (total arrival rate = rho * N * mu)."""
        self.rho = rho
        self.is_sources_set = True

    def set_servers(self, mu: float = 1.0):
        """:param mu: per-server service rate."""
        self.mu = mu
        self.is_servers_set = True

    def _dispatch(self, nq):
        rng = self.generator
        N = self.N
        if self.policy == "random":
            return int(rng.integers(N))
        if self.policy == "jsq":
            return int(np.argmin(nq))
        if self.policy == "jiq":
            idle = np.where(nq == 0)[0]
            return int(rng.choice(idle)) if idle.size else int(rng.integers(N))
        cand = rng.choice(N, size=self.d, replace=False)  # power-of-d
        return int(cand[np.argmin(nq[cand])])

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> QueueResults:
        """Run until `total_served` jobs depart; return the mean sojourn/waiting time."""
        if not (self.is_sources_set and self.is_servers_set):
            raise RuntimeError("sources and servers must be set before run()")
        rng = self.generator
        N, mu = self.N, self.mu
        lam = self.rho * N * mu
        INF = float("inf")

        nq = np.zeros(N, dtype=int)
        fifo = [[] for _ in range(N)]  # arrival times per server (FIFO)
        next_arr = rng.exponential(1 / lam)
        next_dep = np.full(N, INF)
        soj_sum = wait_sum = 0.0
        served = processed = 0
        warm = int(total_served * warmup_fraction)

        while served < total_served + warm:
            dq = int(np.argmin(next_dep))
            if next_arr <= next_dep[dq]:
                t = next_arr
                q = self._dispatch(nq)
                nq[q] += 1
                fifo[q].append(t)
                if nq[q] == 1:
                    next_dep[q] = t + rng.exponential(1 / mu)
                next_arr = t + rng.exponential(1 / lam)
            else:
                t = next_dep[dq]
                a = fifo[dq].pop(0)
                processed += 1
                if processed > warm:
                    soj = t - a
                    soj_sum += soj
                    wait_sum += soj - 1.0 / mu  # approximate wait; exact via start-of-service unneeded for mean
                    served += 1
                nq[dq] -= 1
                next_dep[dq] = t + rng.exponential(1 / mu) if nq[dq] > 0 else INF

        n = max(served, 1)
        return QueueResults(v=[soj_sum / n, 0, 0, 0], w=[max(wait_sum / n, 0.0), 0, 0, 0])
