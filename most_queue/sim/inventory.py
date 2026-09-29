"""
Discrete-event simulator for the M/M/1 queueing-inventory system, (0,S)
policy, backordering (most_queue.theory.inventory.mm1_inventory, EPIC-024).
"""

import time

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import QueueResults


class MM1QueueingInventorySim(BaseSimulationCore):
    """
    (n, i) state, three event types: arrival (n += 1), service completion
    (n -= 1, i -= 1 -- only possible when n >= 1 and i >= 1, server blocked
    otherwise), replenishment (i -> S -- only possible when i == 0).
    """

    def __init__(self, s_max: int, seed: int | None = None):
        super().__init__(seed=seed)
        self.s_max = s_max
        self.l = None
        self.mu = None
        self.theta = None
        self.mean_in_system = None
        self.stockout_prob = None
        self.stock_distribution = None

    def set_sources(self, l: float):
        """:param l: arrival rate."""
        self.l = l

    def set_servers(self, mu: float, theta: float):
        """:param mu: service rate; :param theta: replenishment rate."""
        self.mu, self.theta = mu, theta

    def run(self, total_events: int, warmup_fraction: float = 0.05) -> QueueResults:
        """Run for `total_events` transitions."""
        start = time.process_time()
        rng = self.generator
        t, n, i = 0.0, 0, self.s_max
        warm = int(total_events * warmup_fraction)
        area_n = 0.0
        area_stock = [0.0] * (self.s_max + 1)
        t0 = 0.0

        for step in range(total_events):
            b = self.l
            d = self.mu if (n >= 1 and i >= 1) else 0.0
            r = self.theta if i == 0 else 0.0
            rate = b + d + r
            dt = rng.exponential(1 / rate)
            if step >= warm:
                area_n += n * dt
                area_stock[i] += dt
            else:
                t0 = t + dt
            t += dt

            u = rng.random() * rate
            if u < b:
                n += 1
            elif u < b + d:
                n -= 1
                i -= 1
            else:
                i = self.s_max

        elapsed = t - t0
        self.mean_in_system = area_n / elapsed
        self.stock_distribution = [x / elapsed for x in area_stock]
        self.stockout_prob = self.stock_distribution[0]
        return QueueResults(duration=time.process_time() - start)
