"""
Discrete-event simulator for M/M/1 with deadline-aware admission control
(EPIC-034). Tracks the continuous workload (virtual waiting time) process
U(t) directly -- NOT a discrete "number in system" state, unlike most other
simulators in this library. This is deliberate: an earlier hypothesis that
the number in system n would be a sufficient statistic (giving a simple
state-dependent birth-death chain) was tested and found WRONG -- see
docs/research/llm-serving-deadline-admission-control-2026.md. Validates
most_queue.theory.admission_control.mm1_deadline_admission.
"""

import time

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import AdmissionControlResults


class MM1DeadlineAdmissionControlSim(BaseSimulationCore):
    """
    M/M/1 with deadline-aware admission control, D ~ Exp(theta): an arriving
    job observes the current workload U (PASTA) and is admitted (FCFS) iff
    D > U; otherwise rejected outright (never joins).
    """

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)
        self.l = None
        self.mu = None
        self.theta = None

    def set_sources(self, l: float):
        """:param l: arrival rate."""
        self.l = l

    def set_servers(self, mu: float):
        """:param mu: service rate."""
        self.mu = mu

    def set_deadline(self, theta: float):
        """:param theta: relative-deadline rate, D ~ Exp(theta)."""
        self.theta = theta

    def run(self, total_events: int, warmup_fraction: float = 0.05) -> AdmissionControlResults:
        """Run for `total_events` arrivals."""
        start = time.process_time()
        rng = self.generator
        t, u = 0.0, 0.0
        next_arrival = rng.exponential(1 / self.l)
        warm = int(total_events * warmup_fraction)
        area_u = 0.0
        arrivals = admitted = 0
        v_moments = [0.0, 0.0, 0.0, 0.0]
        t0 = 0.0

        for step in range(total_events):
            dt = next_arrival - t
            u_now = max(u - dt, 0.0)
            if step >= warm:
                # exact trapezoid/triangle integral of the linear decay from u to u_now
                if u <= dt:
                    area_u += u * u / 2.0
                else:
                    area_u += (u + u_now) / 2.0 * dt
            else:
                t0 = next_arrival
            t = next_arrival
            u = u_now

            d = rng.exponential(1 / self.theta)
            if step >= warm:
                arrivals += 1
            if d > u:
                if step >= warm:
                    admitted += 1
                    v = u + rng.exponential(1 / self.mu)
                    power = v
                    for k in range(4):
                        v_moments[k] += power
                        power *= v
                u = u + rng.exponential(1 / self.mu)
            next_arrival = t + rng.exponential(1 / self.l)

        elapsed = t - t0
        self.mean_workload = area_u / elapsed
        self.loss_prob = 1.0 - admitted / arrivals
        v_moments = [m / admitted for m in v_moments]

        result = AdmissionControlResults(v=v_moments, loss_prob=self.loss_prob)
        result.duration = time.process_time() - start
        return result
