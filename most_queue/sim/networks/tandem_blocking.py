"""
Discrete-event simulator for a tandem line with finite buffers and
blocking-after-service (BAS): a job finishing service at node i stays on the
server (blocking it) while node i+1 is full. External arrivals that find
node 1 full are lost. Validates `TandemBlockingCalc`.
"""

import time

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import NetworkMeansResults


class TandemBlockingSim(BaseSimulationCore):
    """
    Tandem with finite buffers, single-server nodes, BAS blocking.

    :param seed: RNG seed.
    """

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)
        self.source = None
        self.arrival_rate = None
        self.servers = None
        self.capacity = None
        self.m = 0
        self.is_sources_set = False
        self.is_nodes_set = False
        self.loss_prob = None
        self.throughput = None

    def set_sources(self, arrival_rate: float, kendall: str = "M"):
        """:param arrival_rate: external rate into node 1 (M — Poisson)."""
        self.arrival_rate = arrival_rate
        self.source = create_distribution(arrival_rate, kendall, self.generator)
        self.is_sources_set = True

    def set_nodes(self, serv_params: list[dict], capacity: list):
        """
        :param serv_params: per-node service distribution, [{"type": kendall, "params": ...}].
        :param capacity: K_i — max jobs at node i (queue + server); None — unlimited.
        """
        self.servers = [create_distribution(p["params"], p["type"], self.generator) for p in serv_params]
        self.capacity = [None if (k is None or k == float("inf")) else int(k) for k in capacity]
        self.m = len(self.capacity)
        self.is_nodes_set = True

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> NetworkMeansResults:
        """
        Run until `total_served` departures from the last node (after warmup).
        """
        start = time.process_time()
        if not (self.is_sources_set and self.is_nodes_set):
            raise RuntimeError("set sources and nodes before run()")

        m = self.m
        inf = float("inf")
        t = 0.0
        jobs = [0] * m  # jobs at node (incl. in service / blocked)
        blocked = [False] * m  # server holds a completed job (BAS)
        completion = [inf] * m
        next_arrival = self.source.generate()

        warm = int(total_served * warmup_fraction)
        served = 0
        stats_on = False
        t0 = 0.0
        area_l = [0.0] * m
        arrived = lost = departures = 0

        def start_service(i):
            completion[i] = t + self.servers[i].generate()

        def try_push(i):
            """Move a completed job from node i to i+1 (or out)."""
            jobs[i] -= 1
            blocked[i] = False
            completion[i] = inf
            if jobs[i] > 0:
                start_service(i)
            if i + 1 < m:
                accept(i + 1)

        def accept(i):
            jobs[i] += 1
            if jobs[i] == 1:
                start_service(i)

        while served < total_served + warm:
            t_next = min([next_arrival, *completion])
            if stats_on:
                dt = t_next - t
                for i in range(m):
                    area_l[i] += jobs[i] * dt
            t = t_next

            if next_arrival <= min(completion):
                if stats_on:
                    arrived += 1
                if self.capacity[0] is not None and jobs[0] >= self.capacity[0]:
                    if stats_on:
                        lost += 1
                else:
                    accept(0)
                next_arrival = t + self.source.generate()
                continue

            i = min(range(m), key=lambda j: completion[j])
            if i + 1 < m and self.capacity[i + 1] is not None and jobs[i + 1] >= self.capacity[i + 1]:
                blocked[i] = True
                completion[i] = inf  # wait for space downstream
            else:
                if i + 1 == m:
                    served += 1
                    if stats_on:
                        departures += 1
                    if not stats_on and served >= warm:
                        stats_on = True
                        t0 = t
                        arrived = lost = departures = 0
                        for j in range(m):
                            area_l[j] = 0.0
                try_push(i)
                # Space freed at node i: unblock upstream servers (cascade)
                j = i
                while j > 0 and blocked[j - 1] and (self.capacity[j] is None or jobs[j] < self.capacity[j]):
                    try_push(j - 1)
                    j -= 1

        elapsed = t - t0
        self.throughput = departures / elapsed
        self.loss_prob = lost / arrived if arrived else 0.0
        mean_jobs = [area_l[i] / elapsed for i in range(m)]

        return NetworkMeansResults(
            v=[sum(mean_jobs) / self.throughput if self.throughput > 0 else 0.0],
            mean_jobs=mean_jobs,
            served=served,
            duration=time.process_time() - start,
        )
