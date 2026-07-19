"""
Discrete-event simulator for the two-class non-preemptive priority
M/M/n + M queue with per-class impatience. Validates
`MMnPriorityImpatienceCalc`.
"""

import time

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import MulticlassResults


class MMnPriorityImpatienceSim(BaseSimulationCore):
    """
    Two priority classes, n exponential servers, per-class abandonment.

    :param n: number of servers.
    :param seed: RNG seed.
    """

    def __init__(self, n: int, seed: int | None = None):
        super().__init__(seed=seed)
        self.n = n
        self.l = None
        self.mu = None
        self.theta = None
        self.mean_queue = None
        self.abandon_probs = None

    def set_sources(self, l: list[float]):
        """:param l: arrival rates [class 0 (priority), class 1]."""
        self.l = [float(x) for x in l]

    def set_servers(self, mu: float, theta: list[float]):
        """:param mu: service rate; :param theta: per-class abandonment rates."""
        self.mu = float(mu)
        self.theta = [float(x) for x in theta]

    def run(self, total_events: int, warmup_fraction: float = 0.05) -> MulticlassResults:
        """
        Rate-based CTMC simulation for `total_events` transitions.
        """
        start = time.process_time()
        rng = self.generator
        n = self.n
        t = 0.0
        busy, q = 0, [0, 0]
        warm = int(total_events * warmup_fraction)
        area_q = [0.0, 0.0]
        arrived = [0, 0]
        abandoned = [0, 0]
        t0 = 0.0

        for step in range(total_events):
            rates = [
                self.l[0],
                self.l[1],
                busy * self.mu,
                q[0] * self.theta[0],
                q[1] * self.theta[1],
            ]
            total_rate = sum(rates)
            dt = rng.exponential(1.0 / total_rate)
            if step >= warm:
                area_q[0] += q[0] * dt
                area_q[1] += q[1] * dt
            else:
                t0 = t + dt
            t += dt
            u = rng.random() * total_rate
            if u < rates[0]:  # class-0 arrival
                if step >= warm:
                    arrived[0] += 1
                if busy < n:
                    busy += 1
                else:
                    q[0] += 1
            elif u < rates[0] + rates[1]:  # class-1 arrival
                if step >= warm:
                    arrived[1] += 1
                if busy < n:
                    busy += 1
                else:
                    q[1] += 1
            elif u < rates[0] + rates[1] + rates[2]:  # service completion
                if q[0] > 0:
                    q[0] -= 1
                elif q[1] > 0:
                    q[1] -= 1
                else:
                    busy -= 1
            elif u < sum(rates[:4]):  # class-0 abandonment
                q[0] -= 1
                if step >= warm:
                    abandoned[0] += 1
            else:  # class-1 abandonment
                q[1] -= 1
                if step >= warm:
                    abandoned[1] += 1

        elapsed = t - t0
        self.mean_queue = [area_q[k] / elapsed for k in range(2)]
        self.abandon_probs = [abandoned[k] / arrived[k] if arrived[k] else 0.0 for k in range(2)]
        w = [[self.mean_queue[k] / self.l[k] if self.l[k] > 0 else 0.0] for k in range(2)]
        return MulticlassResults(w=w, duration=time.process_time() - start)
