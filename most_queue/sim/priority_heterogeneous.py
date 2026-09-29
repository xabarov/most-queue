"""
Discrete-event simulator for M/M/2 with two preemptive priority classes and
heterogeneous servers (most_queue.theory.priority.preemptive.mm2_heterogeneous,
EPIC-025).

Reuses the theory module's four event primitives directly (canonicalize /
class0_arrival / class1_arrival / departure) instead of re-deriving the
dispatch logic -- guarantees the simulated dynamics match the CTMC by
construction, not just by result.
"""

import time

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import PriorityResults
from most_queue.theory.priority.preemptive.mm2_heterogeneous import (
    _class0_arrival,
    _class1_arrival,
    _departure,
    _servers_from_state,
)


class MM2PriorityHeterogeneousSim(BaseSimulationCore):
    """M/M/2, two preemptive priority classes, heterogeneous servers (mu_a != mu_b)."""

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)
        self.l0 = None
        self.l1 = None
        self.mu_a = None
        self.mu_b = None

    def set_sources(self, l: list[float]):
        """:param l: [lambda_0, lambda_1], class 0 = high priority."""
        self.l0, self.l1 = float(l[0]), float(l[1])

    def set_servers(self, mu_a: float, mu_b: float):
        """:param mu_a, mu_b: server rates, any order -- normalised so mu_a >= mu_b."""
        self.mu_a, self.mu_b = max(mu_a, mu_b), min(mu_a, mu_b)

    def run(self, total_events: int, warmup_fraction: float = 0.05) -> PriorityResults:
        """Run for `total_events` transitions."""
        start = time.process_time()
        rng = self.generator
        t = 0.0
        n0, n1, config = 0, 0, None
        warm = int(total_events * warmup_fraction)
        area_n0 = area_n1 = area_in_service_0 = area_in_service_1 = 0.0
        t0 = 0.0

        for step in range(total_events):
            a_class, b_class = _servers_from_state(n0, n1, config)
            rate_a = self.mu_a if a_class is not None else 0.0
            rate_b = self.mu_b if b_class is not None else 0.0
            rate = self.l0 + self.l1 + rate_a + rate_b
            dt = rng.exponential(1 / rate)
            if step >= warm:
                area_n0 += n0 * dt
                area_n1 += n1 * dt
                area_in_service_0 += (a_class, b_class).count(0) * dt
                area_in_service_1 += (a_class, b_class).count(1) * dt
            else:
                t0 = t + dt
            t += dt

            u = rng.random() * rate
            if u < self.l0:
                n0, n1, config = _class0_arrival(n0, n1, a_class, b_class)
            elif u < self.l0 + self.l1:
                n0, n1, config = _class1_arrival(n0, n1, a_class, b_class)
            elif u < self.l0 + self.l1 + rate_a:
                n0, n1, config = _departure(n0, n1, a_class, b_class, "a")
            else:
                n0, n1, config = _departure(n0, n1, a_class, b_class, "b")

        elapsed = t - t0
        e_n0 = area_n0 / elapsed
        e_n1 = area_n1 / elapsed
        e_v0 = e_n0 / self.l0
        e_v1 = e_n1 / self.l1
        e_s0 = (area_in_service_0 / elapsed) / self.l0
        e_s1 = (area_in_service_1 / elapsed) / self.l1

        return PriorityResults(
            v=[[e_v0, 0, 0, 0], [e_v1, 0, 0, 0]],
            w=[[e_v0 - e_s0, 0, 0, 0], [e_v1 - e_s1, 0, 0, 0]],
            duration=time.process_time() - start,
        )
