"""
Simulation of a single-server FCFS queue with server breakdowns and repairs
(preemptive-resume): the server fails at Poisson rate xi while serving; after
a random repair time the interrupted job resumes from where it stopped.
"""

import time

from most_queue.random.utils.create import create_distribution
from most_queue.sim.single_server_disciplines import _SingleServerWorkSim
from most_queue.structs import QueueResults


class UnreliableQueueSim(_SingleServerWorkSim):
    """
    M(GI)/G/1 FCFS with breakdowns while serving (Avi-Itzhak-Naor model).
    Waiting time statistics count the time before the FIRST access to the
    server; the job's "size" for waiting purposes is its completion time
    (service plus its own repairs), matching MG1UnreliableCalc.
    """

    def __init__(self):
        super().__init__()
        self.failure_rate = None
        self.repair_dist = None

    def set_breakdowns(self, xi: float, repair_params, repair_kendall_notation: str):
        """
        Set breakdowns
        :param xi: Poisson failure rate while the server is busy
        :param repair_params: parameters of the repair time distribution
        :param repair_kendall_notation: Kendall notation of the repair time distribution
        """
        if xi < 0:
            raise ValueError(f"Failure rate must be non-negative, got {xi}")
        self.failure_rate = xi
        self.repair_dist = create_distribution(repair_params, repair_kendall_notation, self.generator)

    def _time_to_failure(self) -> float:
        if not self.failure_rate:
            return float("inf")
        return self.generator.exponential(1.0 / self.failure_rate)

    def run(self, total_served: int) -> QueueResults:
        start_time = time.process_time()
        if self.source is None or self.server_dist is None:
            raise ValueError("Set sources and servers first (set_sources/set_servers).")

        queue = []  # arrival times of waiting jobs (FCFS)
        current = None  # [remaining, arrival_time, service_start_time]
        next_arrival = self.source.generate()

        while self.served < total_served:
            if current is None:
                if not queue:
                    self._advance_stats(next_arrival - self.ttek, 0)
                    self.ttek = next_arrival
                    queue.append(self.ttek)
                    next_arrival = self.ttek + self.source.generate()
                arr_time = queue.pop(0)
                current = [self.server_dist.generate(), arr_time, self.ttek]
                continue

            dt_fail = self._time_to_failure()
            dt = min(current[0], dt_fail, next_arrival - self.ttek)

            if dt == next_arrival - self.ttek:
                current[0] -= dt
                self._advance_stats(dt, 1 + len(queue))
                self.ttek = next_arrival
                queue.append(self.ttek)
                next_arrival = self.ttek + self.source.generate()
            elif dt == current[0]:
                self._advance_stats(dt, 1 + len(queue))
                self.ttek += dt
                _, arr_time, service_start = current
                sojourn = self.ttek - arr_time
                # "size" = completion time (service + own repairs)
                self._register_departure(sojourn, self.ttek - service_start)
                current = None
            else:
                # breakdown: serve dt, then repair; arrivals continue during repair
                current[0] -= dt
                self._advance_stats(dt, 1 + len(queue))
                self.ttek += dt
                repair_end = self.ttek + self.repair_dist.generate()
                while next_arrival < repair_end:
                    self._advance_stats(next_arrival - self.ttek, 1 + len(queue))
                    self.ttek = next_arrival
                    queue.append(self.ttek)
                    next_arrival = self.ttek + self.source.generate()
                self._advance_stats(repair_end - self.ttek, 1 + len(queue))
                self.ttek = repair_end

        return self._results(start_time)
