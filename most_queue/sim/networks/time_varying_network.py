"""
Discrete-event simulator for an open Markovian network with a time-varying
(NHPP, thinning-generated) external arrival flow. Collects the phase-bucketed
mean number of jobs in the network across the modulation period. Validates
the PSA calculator (`TimeVaryingNetworkCalc`).
"""

import time

import numpy as np

from most_queue.sim.base_core import BaseSimulationCore


class TimeVaryingNetworkSim(BaseSimulationCore):
    """
    Open network of M/M/n nodes with lambda(t) external arrivals.

    :param period: length of one cycle of lambda(t) (for phase bucketing).
    :param n_buckets: number of phase buckets across the cycle.
    :param seed: RNG seed.
    """

    def __init__(self, period: float, n_buckets: int = 24, seed: int | None = None):
        super().__init__(seed=seed)
        self.period = period
        self.nb = n_buckets
        self.lam_fn = None
        self.lam_max = None
        self.R = None
        self.mu = None
        self.n = None
        self.m = 0
        self.is_sources_set = False
        self.is_nodes_set = False

    def set_sources(self, lam_fn, lam_max: float, R):
        """
        :param lam_fn: callable t -> lambda(t).
        :param lam_max: an upper bound on lambda(t) (thinning envelope).
        :param R: routing matrix, dim (m + 1 x m + 1).
        """
        self.lam_fn = lam_fn
        self.lam_max = lam_max
        self.R = np.asarray(R, dtype=float)
        self.is_sources_set = True

    def set_nodes(self, mu: list, n: list[int]):
        """
        :param mu: exponential service rate per channel at each node.
        :param n: number of channels at each node.
        """
        self.mu = [float(x) for x in mu]
        self.n = [int(x) for x in n]
        self.m = len(self.n)
        self.is_nodes_set = True

    def _route_from(self, row: int) -> int:
        return int(self.generator.choice(self.m + 1, p=self.R[row, : self.m + 1]))

    def run(self, horizon: float, warmup_fraction: float = 0.05):
        """
        Run to time `horizon`. Returns (t_centers, mean_jobs_total) — the
        phase-bucketed time-average number of jobs in the whole network.
        """
        start = time.process_time()
        if not (self.is_sources_set and self.is_nodes_set):
            raise RuntimeError("set sources and nodes before run()")

        rng = self.generator
        inf = float("inf")
        t = 0.0
        completion = [[] for _ in range(self.m)]  # per node: completion times of busy channels
        queue_len = [0] * self.m  # waiting jobs per node
        in_system = 0

        area = [0.0] * self.nb
        span = [0.0] * self.nb
        warm = horizon * warmup_fraction
        next_candidate = rng.exponential(1.0 / self.lam_max)

        def node_accept(i):
            nonlocal in_system
            if len(completion[i]) < self.n[i]:
                completion[i].append(t + rng.exponential(1.0 / self.mu[i]))
            else:
                queue_len[i] += 1

        def collect(dt):
            if t > warm and dt > 0:
                ph = int(((t % self.period) / self.period) * self.nb) % self.nb
                area[ph] += in_system * dt
                span[ph] += dt

        while t < horizon:
            t_comp = min((min(c) for c in completion if c), default=inf)
            t_next = min(next_candidate, t_comp)
            collect(t_next - t)
            t = t_next

            if next_candidate <= t_comp:
                if rng.random() < self.lam_fn(t) / self.lam_max:  # thinning
                    in_system += 1
                    node_accept(self._route_from(0))
                next_candidate = t + rng.exponential(1.0 / self.lam_max)
            else:
                i = min(
                    (j for j in range(self.m) if completion[j]),
                    key=lambda j: min(completion[j]),
                )
                completion[i].remove(min(completion[i]))
                if queue_len[i] > 0:
                    queue_len[i] -= 1
                    completion[i].append(t + rng.exponential(1.0 / self.mu[i]))
                nxt = self._route_from(i + 1)
                if nxt == self.m:
                    in_system -= 1
                else:
                    node_accept(nxt)

        t_centers = [(i + 0.5) / self.nb * self.period for i in range(self.nb)]
        mean_jobs = [area[i] / span[i] if span[i] > 0 else 0.0 for i in range(self.nb)]
        self.time_spent = time.process_time() - start
        return t_centers, mean_jobs
