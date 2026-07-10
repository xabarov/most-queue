"""
Discrete-event simulator for the M/G^[a,b]/1 bulk-service queue.

The server waits until at least `a` customers are queued, then serves a batch of
up to `b` of them together; the whole batch leaves after one (exponential or
general) batch-service time, which may depend on the batch size.
"""

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import QueueResults


class BulkServiceSim(BaseSimulationCore):
    """
    Bulk-service single-server simulator.

    :param a: minimum batch size to start service.
    :param b: maximum batch size.
    :param seed: RNG seed.
    """

    def __init__(self, a: int = 1, b: int = 1, seed: int | None = None):
        super().__init__(seed=seed)
        self.a = a
        self.b = b
        self.source = None
        self.mu_fn = None
        self.batch_kendall = "M"
        self.is_sources_set = False
        self.is_servers_set = False

    def set_sources(self, params, kendall_notation: str = "M"):
        """Set the arrival process."""
        self.source = create_distribution(params, kendall_notation, self.generator)
        self.is_sources_set = True

    def set_servers(self, mu):
        """
        :param mu: batch-service rate — scalar (Exp(mu)) or callable mu(batch_size).
        """
        if callable(mu):
            self.mu_fn = mu
        else:
            self.mu_fn = lambda _i, _mu=float(mu): _mu
        self.is_servers_set = True

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> QueueResults:
        """Run until `total_served` customers have departed; return mean w and v."""
        if not (self.is_sources_set and self.is_servers_set):
            raise RuntimeError("sources and servers must be set before run()")

        INF = float("inf")
        t = 0.0
        next_arrival = self.source.generate()
        queue: list[float] = []  # arrival times of waiting customers (FCFS)
        server_busy = False
        server_done = INF
        batch: list[float] = []  # arrival times of the in-service batch
        batch_start = 0.0

        soj_sum = 0.0
        wait_sum = 0.0
        served = 0
        warmup = int(total_served * warmup_fraction)

        def maybe_start():
            nonlocal server_busy, server_done, batch, batch_start
            if not server_busy and len(queue) >= self.a:
                take = min(self.b, len(queue))
                batch = [queue.pop(0) for _ in range(take)]
                server_busy = True
                batch_start = t
                server_done = t + self.generator.exponential(1.0 / self.mu_fn(take))

        while served < total_served + warmup:
            if next_arrival <= server_done:
                t = next_arrival
                queue.append(t)
                next_arrival = t + self.source.generate()
                maybe_start()
            else:
                t = server_done
                for arr in batch:
                    if served >= warmup:
                        soj_sum += t - arr
                        wait_sum += batch_start - arr
                    served += 1
                server_busy = False
                server_done = INF
                batch = []
                maybe_start()

        n = max(served - warmup, 1)
        return QueueResults(v=[soj_sum / n, 0, 0, 0], w=[wait_sum / n, 0, 0, 0])
