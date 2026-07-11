"""
Discrete-event simulator for an M/G/1 polling system: a single server visits Q
queues cyclically, pays a switchover time between queues, and serves each queue
under the exhaustive or gated discipline. Returns per-queue mean waiting times.
"""

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import MulticlassResults


class PollingSim(BaseSimulationCore):
    """
    :param n_queues: number of queues Q.
    :param discipline: "exhaustive" or "gated".
    :param seed: RNG seed.
    """

    def __init__(self, n_queues: int, discipline: str = "exhaustive", seed: int | None = None):
        super().__init__(seed=seed)
        self.Q = n_queues
        self.discipline = discipline.lower()
        self.sources = None
        self.servers = None
        self.switch = None
        self.is_sources_set = False
        self.is_servers_set = False

    def set_sources(self, lams: list, kendall: str = "M"):
        """:param lams: per-queue arrival rate (Poisson)."""
        self.sources = [create_distribution(l, kendall, self.generator) for l in lams]
        self.is_sources_set = True

    def set_servers(self, params: list, kendall: str = "M"):
        """:param params: per-queue service params (one per queue) for `kendall` distribution."""
        self.servers = [create_distribution(p, kendall, self.generator) for p in params]
        self.is_servers_set = True

    def set_switchover(self, params, kendall: str = "D"):
        """:param params: switchover distribution params (same between every pair of queues)."""
        self.switch = create_distribution(params, kendall, self.generator)

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> MulticlassResults:
        """Run until `total_served` jobs served; return per-queue mean waiting time."""
        if not (self.is_sources_set and self.is_servers_set) or self.switch is None:
            raise RuntimeError("set sources, servers and switchover before run()")
        Q = self.Q
        t = 0.0
        waiting = [[] for _ in range(Q)]  # arrival times waiting at each queue
        next_arr = [self.sources[i].generate() for i in range(Q)]
        wait_sum = [0.0] * Q
        cnt = [0] * Q
        served = 0
        warm = int(total_served * warmup_fraction)
        cur = 0

        def collect(until):
            for i in range(Q):
                while next_arr[i] <= until:
                    waiting[i].append(next_arr[i])
                    next_arr[i] += self.sources[i].generate()

        def serve_one(a):
            nonlocal t, served
            # waiting time = start of service (now) - arrival time
            if served > warm:
                wait_sum[cur] += t - a
                cnt[cur] += 1
            t += self.servers[cur].generate()
            served += 1
            collect(t)

        while served < total_served + warm:
            t += self.switch.generate()  # switchover to queue `cur`
            collect(t)
            if self.discipline == "exhaustive":
                while waiting[cur]:
                    serve_one(waiting[cur].pop(0))
            else:  # gated: only the jobs present at the polling instant
                batch = waiting[cur]
                waiting[cur] = []
                for a in batch:
                    serve_one(a)
            cur = (cur + 1) % Q

        w = [[wait_sum[i] / cnt[i] if cnt[i] else 0.0, 0, 0, 0] for i in range(Q)]
        return MulticlassResults(w=w)
