"""
Discrete-event simulators for the reliability models (unreliable servers):
M/M/c with breakdowns, machine repair problem, M/M/1 working breakdowns,
M/M/1 disasters with repair, M/M/1 retrial with an unreliable server.
All exponential, seeded via `BaseSimulationCore`.
"""

import time

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import QueueResults

INF = float("inf")


class MMcBreakdownsSim(BaseSimulationCore):
    """M/M/c with independent server breakdowns and repairs."""

    def __init__(self, n: int, repairmen: int | None = None, seed: int | None = None):
        super().__init__(seed=seed)
        self.c = n
        self.repairmen = repairmen
        self.l = None
        self.mu = None
        self.xi = None
        self.eta = None
        self.availability = None

    def set_sources(self, l: float):
        """:param l: arrival rate."""
        self.l = l

    def set_servers(self, mu: float, xi: float, eta: float):
        """:param mu: service rate; :param xi: failure rate; :param eta: repair rate."""
        self.mu, self.xi, self.eta = mu, xi, eta

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> QueueResults:
        """Run until `total_served` service completions."""
        start = time.process_time()
        rng = self.generator
        exp = rng.exponential
        t, jobs, up = 0.0, 0, self.c
        served = 0
        warm = int(total_served * warmup_fraction)
        stats_on, t0 = False, 0.0
        area_jobs = area_up = area_busy = 0.0

        next_arr = exp(1 / self.l)
        while served < total_served + warm:
            busy = min(jobs, up)
            rate_srv = busy * self.mu
            rate_fail = up * self.xi
            down = self.c - up
            crew = down if self.repairmen is None else min(down, self.repairmen)
            rate_rep = crew * self.eta
            total_rate = rate_srv + rate_fail + rate_rep
            t_event = t + exp(1 / total_rate) if total_rate > 0 else INF
            t_next = min(next_arr, t_event)
            if stats_on:
                dt = t_next - t
                area_jobs += jobs * dt
                area_up += up * dt
                area_busy += busy * dt
            t = t_next
            if next_arr <= t_event:
                jobs += 1
                next_arr = t + exp(1 / self.l)
                continue
            u = rng.random() * total_rate
            if u < rate_srv:
                jobs -= 1
                served += 1
                if not stats_on and served >= warm:
                    stats_on, t0 = True, t
                    area_jobs = area_up = area_busy = 0.0
            elif u < rate_srv + rate_fail:
                up -= 1
            else:
                up += 1

        elapsed = t - t0
        self.availability = area_up / elapsed / self.c
        return QueueResults(
            v=[area_jobs / elapsed / self.l],
            utilization=area_busy / elapsed / self.c,
            duration=time.process_time() - start,
        )


class MachineRepairSim(BaseSimulationCore):
    """Machine repair problem with warm spares (finite CTMC simulation)."""

    def __init__(self, n_machines: int, n_repairmen: int = 1, n_spares: int = 0, seed: int | None = None):
        super().__init__(seed=seed)
        self.m, self.r, self.s = n_machines, n_repairmen, n_spares
        self.xi = None
        self.eta = None
        self.xi_s = None
        self.availability = None
        self.mean_failed = None

    def set_sources(self, xi: float, eta: float, xi_s: float = 0.0):
        """:param xi: machine failure rate; :param eta: repair rate; :param xi_s: warm-spare failure rate."""
        self.xi, self.eta, self.xi_s = xi, eta, xi_s

    def run(self, total_events: int, warmup_fraction: float = 0.05) -> QueueResults:
        """Run for `total_events` transitions."""
        start = time.process_time()
        rng = self.generator
        t, failed = 0.0, 0
        warm = int(total_events * warmup_fraction)
        area_failed = avail_time = 0.0
        t0 = 0.0

        for step in range(total_events):
            operating = min(self.m, self.m + self.s - failed)
            spares = max(0, self.s - failed)
            b = self.xi * operating + self.xi_s * spares
            d = self.eta * min(failed, self.r)
            rate = b + d
            dt = rng.exponential(1 / rate)
            if step >= warm:
                area_failed += failed * dt
                if failed <= self.s:
                    avail_time += dt
            else:
                t0 = t + dt
            t += dt
            failed += 1 if rng.random() * rate < b else -1

        elapsed = t - t0
        self.availability = avail_time / elapsed
        self.mean_failed = area_failed / elapsed
        return QueueResults(duration=time.process_time() - start)


class MM1WorkingBreakdownsSim(BaseSimulationCore):
    """M/M/1 with working breakdowns (degraded service rate during repair)."""

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)
        self.l = None
        self.mu = None
        self.mu_d = None
        self.xi = None
        self.eta = None
        self.degraded_prob = None

    def set_sources(self, l: float):
        """:param l: arrival rate."""
        self.l = l

    def set_servers(self, mu: float, mu_d: float, xi: float, eta: float):
        """:param mu: normal rate; :param mu_d: degraded rate; :param xi: breakdown; :param eta: repair."""
        self.mu, self.mu_d, self.xi, self.eta = mu, mu_d, xi, eta

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> QueueResults:
        """Run until `total_served` completions."""
        start = time.process_time()
        rng = self.generator
        t, jobs, degraded = 0.0, 0, False
        served = 0
        warm = int(total_served * warmup_fraction)
        stats_on, t0 = False, 0.0
        area_jobs = degraded_time = 0.0

        next_arr = rng.exponential(1 / self.l)
        while served < total_served + warm:
            rate_srv = (self.mu_d if degraded else self.mu) if jobs > 0 else 0.0
            rate_sw = self.eta if degraded else self.xi
            total_rate = rate_srv + rate_sw
            t_event = t + rng.exponential(1 / total_rate) if total_rate > 0 else INF
            t_next = min(next_arr, t_event)
            if stats_on:
                dt = t_next - t
                area_jobs += jobs * dt
                if degraded:
                    degraded_time += dt
            t = t_next
            if next_arr <= t_event:
                jobs += 1
                next_arr = t + rng.exponential(1 / self.l)
                continue
            if rng.random() * total_rate < rate_srv:
                jobs -= 1
                served += 1
                if not stats_on and served >= warm:
                    stats_on, t0 = True, t
                    area_jobs = degraded_time = 0.0
            else:
                degraded = not degraded

        elapsed = t - t0
        self.degraded_prob = degraded_time / elapsed
        return QueueResults(v=[area_jobs / elapsed / self.l], duration=time.process_time() - start)


class MM1DisasterRepairSim(BaseSimulationCore):
    """M/M/1 with disasters (queue flush) followed by exponential repair."""

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)
        self.l = None
        self.mu = None
        self.delta = None
        self.eta = None
        self.down_prob = None

    def set_sources(self, l: float, delta: float):
        """:param l: arrival rate; :param delta: disaster rate."""
        self.l, self.delta = l, delta

    def set_servers(self, mu: float, eta: float):
        """:param mu: service rate; :param eta: repair rate."""
        self.mu, self.eta = mu, eta

    def run(self, total_events: int, warmup_fraction: float = 0.05) -> QueueResults:
        """Run for `total_events` transitions (arrivals + completions + disasters + repairs)."""
        start = time.process_time()
        rng = self.generator
        t, jobs, down = 0.0, 0, False
        warm = int(total_events * warmup_fraction)
        area_jobs = down_time = 0.0
        t0 = 0.0

        for step in range(total_events):
            rate_srv = self.mu if (jobs > 0 and not down) else 0.0
            rate_dis = self.delta if not down else 0.0
            rate_rep = self.eta if down else 0.0
            total_rate = self.l + rate_srv + rate_dis + rate_rep
            dt = rng.exponential(1 / total_rate)
            if step >= warm:
                area_jobs += jobs * dt
                if down:
                    down_time += dt
            else:
                t0 = t + dt
            t += dt
            u = rng.random() * total_rate
            if u < self.l:
                jobs += 1
            elif u < self.l + rate_srv:
                jobs -= 1
            elif u < self.l + rate_srv + rate_dis:
                jobs = 0
                down = True
            else:
                down = False

        elapsed = t - t0
        self.down_prob = down_time / elapsed
        return QueueResults(v=[area_jobs / elapsed / self.l], duration=time.process_time() - start)


class MM1RetrialUnreliableSim(BaseSimulationCore):
    """M/M/1 retrial queue with active breakdowns and repairs."""

    def __init__(self, gamma: float, seed: int | None = None):
        super().__init__(seed=seed)
        self.gamma = gamma
        self.l = None
        self.mu = None
        self.xi = None
        self.eta = None
        self.availability = None
        self.mean_orbit = None

    def set_sources(self, l: float):
        """:param l: arrival rate."""
        self.l = l

    def set_servers(self, mu: float, xi: float = 0.0, eta: float = 1.0):
        """:param mu: service rate; :param xi: active failure rate; :param eta: repair rate."""
        self.mu, self.xi, self.eta = mu, xi, eta

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> QueueResults:
        """Run until `total_served` completions."""
        start = time.process_time()
        rng = self.generator
        IDLE, BUSY, DOWN = 0, 1, 2
        t, orbit, state = 0.0, 0, IDLE
        served = 0
        warm = int(total_served * warmup_fraction)
        stats_on, t0 = False, 0.0
        area_orbit = busy_time = down_time = 0.0

        while served < total_served + warm:
            rate_retr = orbit * self.gamma if state == IDLE else 0.0
            rate_srv = self.mu if state == BUSY else 0.0
            rate_fail = self.xi if state == BUSY else 0.0
            rate_rep = self.eta if state == DOWN else 0.0
            total_rate = self.l + rate_retr + rate_srv + rate_fail + rate_rep
            dt = rng.exponential(1 / total_rate)
            if stats_on:
                area_orbit += orbit * dt
                if state == BUSY:
                    busy_time += dt
                elif state == DOWN:
                    down_time += dt
            t += dt
            u = rng.random() * total_rate
            if u < self.l:  # new arrival
                if state == IDLE:
                    state = BUSY
                else:
                    orbit += 1
            elif u < self.l + rate_retr:  # successful retrial
                orbit -= 1
                state = BUSY
            elif u < self.l + rate_retr + rate_srv:  # completion
                state = IDLE
                served += 1
                if not stats_on and served >= warm:
                    stats_on, t0 = True, t
                    area_orbit = busy_time = down_time = 0.0
            elif u < self.l + rate_retr + rate_srv + rate_fail:  # breakdown
                orbit += 1
                state = DOWN
            else:  # repair
                state = IDLE

        elapsed = t - t0
        self.mean_orbit = area_orbit / elapsed
        self.availability = 1.0 - down_time / elapsed
        mean_in_system = self.mean_orbit + busy_time / elapsed
        return QueueResults(v=[mean_in_system / self.l], duration=time.process_time() - start)
