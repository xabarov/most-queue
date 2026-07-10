"""
Discrete-event simulator for the multiserver-job (MSJ) model.

Each job belongs to a class with a fixed **server need** (number of servers it
occupies simultaneously for its whole service) and an exponential service rate.
There are k servers. Scheduling is FCFS with head-of-line blocking: the servers
are filled by the oldest jobs greedily, and the first job that does not fit stops
the scan (later jobs do not jump ahead), which is the canonical MSJ discipline
whose mean response time is the subject of current research.
"""

from dataclasses import dataclass

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import QueueResults


@dataclass
class MsjClass:
    """A multiserver-job class."""

    arrival_rate: float
    servers: int  # server need
    mu: float  # service rate (exponential)


class _Job:
    __slots__ = ("cls", "arr", "done")

    def __init__(self, cls, arr):
        self.cls = cls
        self.arr = arr
        self.done = float("inf")  # scheduled completion time (inf = not started)


class MsjSim(BaseSimulationCore):
    """
    FCFS multiserver-job simulator.

    :param k: number of servers.
    :param classes: list of MsjClass.
    :param seed: RNG seed.
    """

    def __init__(self, k: int, classes: list[MsjClass], seed: int | None = None):
        super().__init__(seed=seed)
        self.k = k
        self.classes = classes
        self.is_sources_set = True
        self.is_servers_set = True

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> QueueResults:
        """Run until `total_served` jobs depart; return per-class and overall mean sojourn."""
        rng = self.generator
        m = len(self.classes)
        rates = [c.arrival_rate for c in self.classes]
        total_rate = sum(rates)
        INF = float("inf")

        t = 0.0
        next_arrival = rng.exponential(1.0 / total_rate)
        jobs: list[_Job] = []  # in arrival order (FCFS)

        soj_sum = [0.0] * m
        soj_cnt = [0] * m
        served = 0
        warmup = int(total_served * warmup_fraction)

        def schedule():
            """Greedy FCFS fill: start the maximal fitting prefix; block at first misfit."""
            free = self.k
            for job in jobs:
                need = self.classes[job.cls].servers
                if job.done < INF:  # already in service
                    free -= need
                    continue
                if need <= free:
                    free -= need
                    job.done = t + rng.exponential(1.0 / self.classes[job.cls].mu)
                else:
                    break  # FCFS head-of-line blocking

        while served < total_served + warmup:
            # earliest completion among in-service jobs
            done_idx, done_t = -1, INF
            for idx, job in enumerate(jobs):
                if job.done < done_t:
                    done_t, done_idx = job.done, idx

            if next_arrival <= done_t:
                t = next_arrival
                # pick class
                u = rng.random() * total_rate
                acc, cls = 0.0, 0
                for i in range(m):
                    acc += rates[i]
                    if u <= acc:
                        cls = i
                        break
                jobs.append(_Job(cls, t))
                next_arrival = t + rng.exponential(1.0 / total_rate)
                schedule()
            else:
                t = done_t
                job = jobs.pop(done_idx)
                if served >= warmup:
                    soj_sum[job.cls] += t - job.arr
                    soj_cnt[job.cls] += 1
                served += 1
                schedule()

        v_per_class = [soj_sum[i] / soj_cnt[i] if soj_cnt[i] else 0.0 for i in range(m)]
        total_cnt = sum(soj_cnt)
        v_overall = sum(soj_sum) / total_cnt if total_cnt else 0.0
        res = QueueResults(v=[v_overall, 0, 0, 0])
        res.v_per_class = v_per_class  # type: ignore[attr-defined]
        return res
