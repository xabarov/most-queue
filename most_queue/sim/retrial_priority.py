"""
Discrete-event simulator for the two-class M/M/1 retrial queue with a
priority class (waits in queue) and an ordinary class (joins the orbit,
retries at rate gamma each). Validates `MM1RetrialPriorityCalc`.
"""

import time

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import MulticlassResults

IDLE, SERV0, SERV1 = 0, 1, 2


class MM1RetrialPrioritySim(BaseSimulationCore):
    """
    Two-class retrial queue with priority: rate-based CTMC simulation.

    :param gamma: retrial rate of each orbiting customer.
    :param seed: RNG seed.
    """

    def __init__(self, gamma: float, seed: int | None = None):
        super().__init__(seed=seed)
        self.gamma = float(gamma)
        self.l = None
        self.mu = None
        self.mean_priority_queue = None
        self.mean_orbit = None

    def set_sources(self, l: list[float]):
        """:param l: arrival rates [class 0 (priority), class 1 (orbiting)]."""
        self.l = [float(x) for x in l]

    def set_servers(self, mu: list[float]):
        """:param mu: service rates per class."""
        self.mu = [float(x) for x in mu]

    def run(self, total_events: int, warmup_fraction: float = 0.05) -> MulticlassResults:
        """
        Run for `total_events` transitions.
        """
        start = time.process_time()
        rng = self.generator
        t = 0.0
        q, orbit, state = 0, 0, IDLE
        warm = int(total_events * warmup_fraction)
        area_q = area_orbit = 0.0
        p_serv = [0.0, 0.0]
        t0 = 0.0

        for step in range(total_events):
            rates = [
                self.l[0],
                self.l[1],
                self.mu[0] if state == SERV0 else 0.0,
                self.mu[1] if state == SERV1 else 0.0,
                orbit * self.gamma if state == IDLE else 0.0,
            ]
            total_rate = sum(rates)
            dt = rng.exponential(1.0 / total_rate)
            if step >= warm:
                area_q += q * dt
                area_orbit += orbit * dt
                if state == SERV0:
                    p_serv[0] += dt
                elif state == SERV1:
                    p_serv[1] += dt
            else:
                t0 = t + dt
            t += dt
            u = rng.random() * total_rate
            if u < rates[0]:  # priority arrival
                if state == IDLE:
                    state = SERV0
                else:
                    q += 1
            elif u < rates[0] + rates[1]:  # ordinary arrival
                if state == IDLE:
                    state = SERV1
                else:
                    orbit += 1
            elif u < rates[0] + rates[1] + rates[2] + rates[3]:  # completion
                if q > 0:
                    q -= 1
                    state = SERV0
                else:
                    state = IDLE
            else:  # successful retrial
                orbit -= 1
                state = SERV1

        elapsed = t - t0
        self.mean_priority_queue = area_q / elapsed
        self.mean_orbit = area_orbit / elapsed
        w = [
            [self.mean_priority_queue / self.l[0] if self.l[0] > 0 else 0.0],
            [self.mean_orbit / self.l[1] if self.l[1] > 0 else 0.0],
        ]
        v = [
            [(self.mean_priority_queue + p_serv[0] / elapsed) / self.l[0] if self.l[0] > 0 else 0.0],
            [(self.mean_orbit + p_serv[1] / elapsed) / self.l[1] if self.l[1] > 0 else 0.0],
        ]
        return MulticlassResults(w=w, v=v, duration=time.process_time() - start)
