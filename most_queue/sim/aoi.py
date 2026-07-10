"""
Discrete-event simulator for Age of Information (AoI).

Tracks the sawtooth age process Delta(t) = t - u(t), where u(t) is the
generation time of the freshest *delivered* update, and reports the time-average
AoI and the average peak AoI (PAoI). Supports FCFS, LCFS (non-preemptive) and
LCFS-preemptive (a fresh arrival preempts and discards the in-service update)
disciplines on one or more servers.
"""

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import AoIResults


class AoISim(BaseSimulationCore):
    """
    Age-of-Information simulator.

    :param num_channels: number of servers.
    :param discipline: "FCFS", "LCFS" (non-preemptive) or "LCFS-PR" (preemptive:
        a new update preempts and discards the one in service).
    :param seed: RNG seed for reproducible runs.
    """

    def __init__(self, num_channels: int = 1, discipline: str = "FCFS", seed: int | None = None):
        super().__init__(seed=seed)
        self.n = num_channels
        self.discipline = discipline.upper()
        if self.discipline == "LCFS-PR" and num_channels != 1:
            raise ValueError("LCFS-PR is only supported for a single server")
        self.source = None
        self.server = None
        self.is_sources_set = False
        self.is_servers_set = False

    def set_sources(self, params, kendall_notation: str = "M"):
        """Set the arrival (update generation) process."""
        self.source = create_distribution(params, kendall_notation, self.generator)
        self.is_sources_set = True

    def set_servers(self, params, kendall_notation: str = "M"):
        """Set the service (update delivery) process."""
        self.server = create_distribution(params, kendall_notation, self.generator)
        self.is_servers_set = True

    def run(self, num_updates: int, warmup_fraction: float = 0.05) -> AoIResults:
        """
        Run until `num_updates` fresh deliveries have been counted after warm-up.

        :param warmup_fraction: initial fraction of deliveries discarded before
            age accounting begins (removes transient bias).
        """
        if not (self.is_sources_set and self.is_servers_set):
            raise RuntimeError("sources and servers must be set before run()")

        INF = float("inf")
        t = 0.0
        next_arrival = self.source.generate()
        server_done = [INF] * self.n  # completion time per server
        server_gen = [0.0] * self.n  # generation time of the update in service
        queue: list[float] = []  # generation times of waiting updates (arrival order)

        freshest_gen = 0.0  # generation time of the freshest delivered update
        area = 0.0  # integral of the age over the accounting window
        peak_sum = 0.0
        delivered = 0
        counted = 0  # fresh deliveries counted after warm-up
        accounting = False
        acct_start_t = 0.0
        last_t = 0.0
        warmup = int(num_updates * warmup_fraction)
        target = num_updates + warmup

        def free_server():
            for i in range(self.n):
                if server_done[i] == INF:
                    return i
            return -1

        def start(gen_time):
            i = free_server()
            server_gen[i] = gen_time
            server_done[i] = t + self.server.generate()

        while delivered < target:
            done_idx = min(range(self.n), key=lambda i: server_done[i])
            next_done = server_done[done_idx]

            if next_arrival <= next_done:
                # ---- arrival of a new update ----
                t = next_arrival
                next_arrival = t + self.source.generate()
                gen = t
                if self.discipline == "LCFS-PR":
                    server_gen[0] = gen
                    server_done[0] = t + self.server.generate()
                elif free_server() != -1:
                    start(gen)
                else:
                    queue.append(gen)
            else:
                # ---- delivery of an update ----
                t = next_done
                gen = server_gen[done_idx]
                server_done[done_idx] = INF
                delivered += 1
                is_fresh = gen > freshest_gen

                if accounting and is_fresh:
                    # age grew as (t - freshest_gen) then resets: accumulate the
                    # trapezoid and record the peak (age just before the reset)
                    area += ((t - freshest_gen) ** 2 - (last_t - freshest_gen) ** 2) / 2.0
                    peak_sum += t - freshest_gen
                    last_t = t
                    counted += 1
                if is_fresh:
                    freshest_gen = gen

                if not accounting and delivered >= warmup:
                    accounting = True
                    acct_start_t = t
                    last_t = t
                    freshest_gen = gen

                if queue:
                    nxt = queue.pop() if self.discipline == "LCFS" else queue.pop(0)
                    start(nxt)

        # area is accumulated up to the last fresh delivery (last_t); the window
        # is [acct_start_t, last_t]
        window = max(last_t - acct_start_t, 1e-12)
        return AoIResults(avg_aoi=area / window, peak_aoi=peak_sum / max(counted, 1))
