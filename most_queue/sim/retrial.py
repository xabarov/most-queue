"""
Simulation of a single-server retrial queue (classical linear retrial policy):
a job that finds the server busy joins the orbit and retries after an
exponential time with rate gamma; with j jobs in orbit the total retrial rate
is j*gamma (memoryless, so a random orbiting job succeeds).
"""

import time

from most_queue.sim.single_server_disciplines import _SingleServerWorkSim
from most_queue.structs import QueueResults


class RetrialQueueSim(_SingleServerWorkSim):
    """
    M(GI)/G/1 retrial queue simulator. Waiting time = time spent in the orbit
    (zero for jobs that find the server idle on arrival); the number of jobs
    in the system = orbit size + the job in service.
    """

    def __init__(self, gamma: float):
        """
        :param gamma: retrial rate of each orbiting job
        """
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"Retrial rate gamma must be positive, got {gamma}")
        self.gamma = gamma
        self._current_wait = 0.0  # orbit time of the job currently in service

    def run(self, total_served: int) -> QueueResults:
        start_time = time.process_time()
        if self.source is None or self.server_dist is None:
            raise ValueError("Set sources and servers first (set_sources/set_servers).")

        rng = self.generator
        orbit = []  # arrival times of orbiting jobs
        service_end = None  # completion time of the job in service
        current_arr = None  # arrival time of the job in service
        next_arrival = self.source.generate()

        while self.served < total_served:
            in_sys = len(orbit) + (1 if service_end is not None else 0)

            if service_end is None:
                # server idle: next event is an arrival or a successful retry
                next_retry = self.ttek + rng.exponential(1.0 / (len(orbit) * self.gamma)) if orbit else float("inf")
                if next_arrival <= next_retry:
                    self._advance_stats(next_arrival - self.ttek, in_sys)
                    self.ttek = next_arrival
                    current_arr = self.ttek
                    self._current_wait = 0.0
                    service_end = self.ttek + self.server_dist.generate()
                    next_arrival = self.ttek + self.source.generate()
                else:
                    self._advance_stats(next_retry - self.ttek, in_sys)
                    self.ttek = next_retry
                    idx = int(rng.integers(len(orbit)))
                    current_arr = orbit.pop(idx)
                    self._current_wait = self.ttek - current_arr
                    service_end = self.ttek + self.server_dist.generate()
            else:
                # server busy: retries fail, next event is an arrival or completion
                if next_arrival <= service_end:
                    self._advance_stats(next_arrival - self.ttek, in_sys)
                    self.ttek = next_arrival
                    orbit.append(self.ttek)
                    next_arrival = self.ttek + self.source.generate()
                else:
                    self._advance_stats(service_end - self.ttek, in_sys)
                    self.ttek = service_end
                    sojourn = self.ttek - current_arr
                    # size = sojourn - orbit wait, so _register_departure stores
                    # waiting = time in orbit
                    self._register_departure(sojourn, sojourn - self._current_wait)
                    service_end = None
                    current_arr = None

        return self._results(start_time)
