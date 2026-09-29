"""
Discrete-event simulator for the M/M/1 queueing-inventory system, (0,S)
policy, backordering (most_queue.theory.inventory.mm1_inventory, EPIC-024) or
lost sales (EPIC-026).
"""

import time
from typing import Literal

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import QueueResults

Policy = Literal["backorder", "lost_sales"]


class MM1QueueingInventorySim(BaseSimulationCore):
    """
    (n, i) state, three event types: arrival (n += 1, or lost if i == 0 under
    policy="lost_sales"), service completion (n -= 1, i -= 1 -- only possible
    when n >= 1 and i >= 1, server blocked otherwise), replenishment
    (i -> S -- only possible when i == 0).
    """

    def __init__(self, s_max: int, policy: Policy = "backorder", seed: int | None = None):
        super().__init__(seed=seed)
        self.s_max = s_max
        self.policy: Policy = policy
        self.l = None
        self.mu = None
        self.theta = None
        self.mean_in_system = None
        self.stockout_prob = None
        self.stock_distribution = None
        self.loss_prob = None

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
        arrivals = lost = 0
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
                if self.policy == "lost_sales" and i == 0:
                    if step >= warm:
                        lost += 1
                else:
                    n += 1
                if step >= warm:
                    arrivals += 1
            elif u < b + d:
                n -= 1
                i -= 1
            else:
                i = self.s_max

        elapsed = t - t0
        self.mean_in_system = area_n / elapsed
        self.stock_distribution = [x / elapsed for x in area_stock]
        self.stockout_prob = self.stock_distribution[0]
        self.loss_prob = (lost / arrivals) if self.policy == "lost_sales" else 0.0
        return QueueResults(duration=time.process_time() - start)
