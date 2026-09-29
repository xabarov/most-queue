"""
Discrete-event simulator for the M/M/1 queueing-inventory system, (s,S)
policy (EPIC-027), backordering (most_queue.theory.inventory.mm1_inventory,
EPIC-024) or lost sales (EPIC-026).
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
    (i -> S -- only possible when i <= s, the reorder point).
    """

    def __init__(self, s_max: int, s: int = 0, policy: Policy = "backorder", seed: int | None = None):
        super().__init__(seed=seed)
        if not 0 <= s < s_max:
            raise ValueError(f"reorder point s must satisfy 0 <= s < s_max, got s={s}, s_max={s_max}")
        self.s_max = s_max
        self.s = s
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
            r = self.theta if i <= self.s else 0.0
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


class MM2QueueingInventoryHeterogeneousSim(BaseSimulationCore):
    """
    M/M/2 queueing-inventory simulator with two HETEROGENEOUS servers
    (EPIC-033): unlike MMcQueueingInventorySim, which server is busy matters
    (rates differ), so it must be tracked explicitly.

    State: (n, i, config), config in {1, 2} tracks which single server is
    busy when n == 1 (needed since the two servers' rates differ); for n=0
    nobody is busy, for n>=2 both are busy -- config is only meaningful (and
    only updated) at n == 1, mirroring
    theory.inventory.mm2_heterogeneous_inventory's state design exactly.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, s_max: int, s: int = 0, policy: Policy = "backorder", seed: int | None = None
    ):
        super().__init__(seed=seed)
        if not 0 <= s < s_max:
            raise ValueError(f"reorder point s must satisfy 0 <= s < s_max, got s={s}, s_max={s_max}")
        self.s_max = s_max
        self.s = s
        self.policy: Policy = policy
        self.l = None
        self.mu1 = None
        self.mu2 = None
        self.theta = None
        self.mean_in_system = None
        self.stockout_prob = None
        self.stock_distribution = None
        self.loss_prob = None

    def set_sources(self, l: float):
        """:param l: arrival rate."""
        self.l = l

    def set_servers(self, mu1: float, mu2: float, theta: float):
        """:param mu1: server 1 rate; :param mu2: server 2 rate; :param theta: replenishment rate."""
        self.mu1, self.mu2, self.theta = mu1, mu2, theta

    def run(self, total_events: int, warmup_fraction: float = 0.05) -> QueueResults:  # pylint: disable=too-many-locals
        """Run for `total_events` transitions."""
        start = time.process_time()
        rng = self.generator
        t, n, i = 0.0, 0, self.s_max
        config = None  # which server is busy, only meaningful at n == 1
        warm = int(total_events * warmup_fraction)
        area_n = 0.0
        area_stock = [0.0] * (self.s_max + 1)
        arrivals = lost = 0
        t0 = 0.0

        for step in range(total_events):
            b = self.l
            if i >= 1:
                d = (self.mu1 if config == 1 else self.mu2) if n == 1 else (self.mu1 + self.mu2 if n >= 2 else 0.0)
            else:
                d = 0.0
            r = self.theta if i <= self.s else 0.0
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
                    if n == 0:
                        config = 1  # tie-break convention: assign to server 1 first
                    n += 1
                if step >= warm:
                    arrivals += 1
            elif u < b + d:
                if n == 1:
                    n = 0
                    i -= 1
                    config = None
                else:  # n >= 2: race between the two servers for which one just finished
                    server1_finished = rng.random() * (self.mu1 + self.mu2) < self.mu1
                    n -= 1
                    i -= 1
                    if n == 1:
                        config = 2 if server1_finished else 1  # the survivor
            else:
                i = self.s_max

        elapsed = t - t0
        self.mean_in_system = area_n / elapsed
        self.stock_distribution = [x / elapsed for x in area_stock]
        self.stockout_prob = self.stock_distribution[0]
        self.loss_prob = (lost / arrivals) if self.policy == "lost_sales" else 0.0
        return QueueResults(duration=time.process_time() - start)


class MMcQueueingInventorySim(BaseSimulationCore):
    """
    M/M/c queueing-inventory simulator (EPIC-028): identical to
    MM1QueueingInventorySim except the service rate is min(n,c)*mu instead of
    mu -- the number of busy servers is min(n,c), a deterministic function of
    the number of customers n, so no extra state is needed.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, c: int, s_max: int, s: int = 0, policy: Policy = "backorder", seed: int | None = None
    ):
        super().__init__(seed=seed)
        if not 0 <= s < s_max:
            raise ValueError(f"reorder point s must satisfy 0 <= s < s_max, got s={s}, s_max={s_max}")
        self.c = c
        self.s_max = s_max
        self.s = s
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
        """:param mu: per-server service rate; :param theta: replenishment rate."""
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
            d = min(n, self.c) * self.mu if i >= 1 else 0.0
            r = self.theta if i <= self.s else 0.0
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
