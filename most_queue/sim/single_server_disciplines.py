"""
Compact single-server simulators for work-conserving disciplines that the
event/channel engine of QsSim does not cover: egalitarian Processor Sharing
and preemptive-resume LCFS. Both track remaining work of every job in the
system directly.
"""

import time

import numpy as np

from most_queue.random.utils.create import create_distribution
from most_queue.structs import QueueResults

NUM_OF_MOMENTS = 4
P_NUM = 100


class _SingleServerWorkSim:
    """
    Common machinery: sources/servers setup, statistics collection
    (sojourn/waiting raw moments, time-weighted state probabilities).
    """

    def __init__(self):
        self.generator = np.random.default_rng()
        self.source = None
        self.server_dist = None

        self.ttek = 0.0
        self.served = 0
        self.busy_time = 0.0

        self.v = [0.0] * NUM_OF_MOMENTS
        self.w = [0.0] * NUM_OF_MOMENTS
        self.p_time = [0.0] * P_NUM

    def set_sources(self, params, kendall_notation: str):
        """
        Set the interarrival time distribution (see create_distribution).
        """
        self.source = create_distribution(params, kendall_notation, self.generator)

    def set_servers(self, params, kendall_notation: str):
        """
        Set the service time (job size) distribution (see create_distribution).
        """
        self.server_dist = create_distribution(params, kendall_notation, self.generator)

    def _advance_stats(self, dt: float, jobs_in_system: int):
        if jobs_in_system < P_NUM:
            self.p_time[jobs_in_system] += dt
        if jobs_in_system > 0:
            self.busy_time += dt

    def _register_departure(self, sojourn: float, size: float):
        self.served += 1
        waiting = sojourn - size
        for k in range(NUM_OF_MOMENTS):
            self.v[k] += sojourn ** (k + 1)
            self.w[k] += waiting ** (k + 1)

    def _results(self, start_time: float) -> QueueResults:
        v = [m / self.served for m in self.v]
        w = [m / self.served for m in self.w]
        p = [t / self.ttek for t in self.p_time]
        utilization = self.busy_time / self.ttek
        return QueueResults(v=v, w=w, p=p, utilization=utilization, duration=time.process_time() - start_time)

    def run(self, total_served: int) -> QueueResults:
        """
        Run simulation until total_served jobs have departed.
        """
        raise NotImplementedError


class ProcessorSharingSim(_SingleServerWorkSim):
    """
    M(GI)/G/1 with egalitarian Processor Sharing: each of the k jobs present
    receives service at rate 1/k.
    """

    def run(self, total_served: int) -> QueueResults:
        start_time = time.process_time()
        if self.source is None or self.server_dist is None:
            raise ValueError("Set sources and servers first (set_sources/set_servers).")

        jobs = []  # [remaining, arrival_time, size]
        next_arrival = self.source.generate()

        while self.served < total_served:
            k = len(jobs)
            if k == 0:
                self._advance_stats(next_arrival - self.ttek, 0)
                self.ttek = next_arrival
                jobs.append([self.server_dist.generate(), self.ttek, 0.0])
                jobs[-1][2] = jobs[-1][0]
                next_arrival = self.ttek + self.source.generate()
                continue

            min_rem = min(job[0] for job in jobs)
            completion_time = self.ttek + min_rem * k

            if next_arrival < completion_time:
                dt = next_arrival - self.ttek
                for job in jobs:
                    job[0] -= dt / k
                self._advance_stats(dt, k)
                self.ttek = next_arrival
                size = self.server_dist.generate()
                jobs.append([size, self.ttek, size])
                next_arrival = self.ttek + self.source.generate()
            else:
                dt = completion_time - self.ttek
                for job in jobs:
                    job[0] -= min_rem
                self._advance_stats(dt, k)
                self.ttek = completion_time
                done_idx = min(range(k), key=lambda i: jobs[i][0])
                _, arr_time, size = jobs.pop(done_idx)
                self._register_departure(self.ttek - arr_time, size)

        return self._results(start_time)


class LcfsPRSim(_SingleServerWorkSim):
    """
    M(GI)/G/1 with preemptive-resume LCFS: a new arrival preempts the job in
    service; interrupted jobs are stacked and resumed from where they stopped.
    """

    def run(self, total_served: int) -> QueueResults:
        start_time = time.process_time()
        if self.source is None or self.server_dist is None:
            raise ValueError("Set sources and servers first (set_sources/set_servers).")

        stack = []  # [remaining, arrival_time, size]; last element is in service
        next_arrival = self.source.generate()

        while self.served < total_served:
            if not stack:
                self._advance_stats(next_arrival - self.ttek, 0)
                self.ttek = next_arrival
                size = self.server_dist.generate()
                stack.append([size, self.ttek, size])
                next_arrival = self.ttek + self.source.generate()
                continue

            current = stack[-1]
            completion_time = self.ttek + current[0]

            if next_arrival < completion_time:
                dt = next_arrival - self.ttek
                current[0] -= dt
                self._advance_stats(dt, len(stack))
                self.ttek = next_arrival
                size = self.server_dist.generate()
                stack.append([size, self.ttek, size])
                next_arrival = self.ttek + self.source.generate()
            else:
                dt = completion_time - self.ttek
                self._advance_stats(dt, len(stack))
                self.ttek = completion_time
                _, arr_time, size = stack.pop()
                self._register_departure(self.ttek - arr_time, size)

        return self._results(start_time)
