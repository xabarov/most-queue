"""
Discrete-event simulator for the M/M/c queue whose service rate depends on the
queueing delay the customer experienced (D'Auria, Adan, Bekker & Kulkarni,
EJOR 299(2):566-579, 2022). Validates
``most_queue.theory.delay_dependent.mmc_threshold_service``.

Like ``most_queue.sim.admission_control``, and unlike most simulators here, this
one tracks the continuous virtual queueing time rather than a discrete number in
system -- that is the quantity the model is defined on, and the number in system
is not a sufficient statistic once the service rate depends on the delay.

Under FCFS with ``c`` identical servers the arriving customer always takes the
server that frees up first, so the whole system state is the multiset of the
``c`` next-free times. The VQT seen by an arrival at ``t`` is
``max(0, min(free_times) - t)``, which by PASTA has the stationary distribution
of the VQT process the theory computes.
"""

import heapq
import time

import numpy as np

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import QueueResults


class MMcDelayDependentServiceSim(BaseSimulationCore):
    """
    M/M/c in which a customer whose queueing delay is at most ``k`` is served at
    rate ``mu1``, and one that waited longer at rate ``mu2``.

    :param c: number of identical servers.
    :param k: delay threshold separating the two service rates.
    :param seed: RNG seed.
    """

    def __init__(self, c: int, k: float, seed: int | None = None):
        super().__init__(seed=seed)
        if c < 1:
            raise ValueError(f"c must be >= 1, got {c}")
        if k < 0:
            raise ValueError(f"threshold k must be non-negative, got {k}")
        self.c = c
        self.k = float(k)
        self.l: float | None = None
        self.mu1: float | None = None
        self.mu2: float | None = None
        self.waits: np.ndarray | None = None
        self.sojourns: np.ndarray | None = None

    def set_sources(self, l: float):
        """:param l: arrival rate (Poisson)."""
        self.l = float(l)

    def set_servers(self, mu1: float, mu2: float):
        """
        :param mu1: service rate for a customer whose delay was ``<= k``.
        :param mu2: service rate for a customer whose delay was ``> k``.
        """
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)

    def run(self, total_served: int, warmup_fraction: float = 0.2) -> QueueResults:
        """
        Run for ``total_served`` arrivals.

        :param warmup_fraction: leading fraction of customers discarded. The run
            starts empty, which biases early delays down; the default is
            deliberately generous because the VQT of a multi-server queue is
            strongly autocorrelated and settles slowly.
        """
        start = time.process_time()
        rng = self.generator
        free = [0.0] * self.c  # next-free time of each server (a min-heap)
        heapq.heapify(free)

        waits = np.empty(total_served)
        sojourns = np.empty(total_served)
        t = 0.0
        for i in range(total_served):
            t += rng.exponential(1.0 / self.l)
            earliest = heapq.heappop(free)
            service_start = max(t, earliest)
            wait = service_start - t
            rate = self.mu1 if wait <= self.k else self.mu2
            duration = rng.exponential(1.0 / rate)
            heapq.heappush(free, service_start + duration)
            waits[i] = wait
            sojourns[i] = wait + duration

        warm = int(total_served * warmup_fraction)
        self.waits = waits[warm:]
        self.sojourns = sojourns[warm:]

        result = QueueResults(
            v=[float(self.sojourns.mean())],
            w=[float(self.waits.mean())],
            p=None,
            utilization=self.l * float((self.sojourns - self.waits).mean()) / self.c,
        )
        result.duration = time.process_time() - start
        return result

    def get_p0_wait(self) -> float:
        """Empirical ``P(W = 0)`` -- the fraction that found a server free."""
        return float((self.waits == 0.0).mean())

    def get_cdf(self, x: float) -> float:
        """Empirical ``P(W <= x)``."""
        return float((self.waits <= x).mean())

    def get_tail(self, x: float) -> float:
        """Empirical ``P(W > x)``."""
        return float((self.waits > x).mean())
