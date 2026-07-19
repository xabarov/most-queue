"""
Discrete-event simulator for the M/G/1 accumulating priority queue: at every
service completion the waiting customer with the largest accumulated priority
rate_k * (waiting time) is served next (non-preemptive, linear accumulation).
Validates `MG1AccumulatingPriorityCalc`.
"""

import time

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import MulticlassResults


class AccumulatingPrioritySim(BaseSimulationCore):
    """
    M/G/1 with linear accumulating priorities (class 0 — highest rate).

    :param seed: RNG seed.
    """

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)
        self.sources = None
        self.servers = None
        self.rates = None
        self.n_classes = 0
        self.is_sources_set = False
        self.is_servers_set = False

    def set_sources(self, l: list[float]):
        """:param l: Poisson arrival rate per class."""
        self.sources = [create_distribution(rate, "M", self.generator) for rate in l]
        self.n_classes = len(l)
        self.is_sources_set = True

    def set_servers(self, serv_params: list[dict], rates: list[float]):
        """
        :param serv_params: per-class service spec, [{"type": kendall, "params": ...}].
        :param rates: accumulation rate per class.
        """
        self.servers = [create_distribution(p["params"], p["type"], self.generator) for p in serv_params]
        self.rates = [float(r) for r in rates]
        self.is_servers_set = True

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> MulticlassResults:
        """
        Run until `total_served` completions; returns per-class mean waiting
        and sojourn times (first moments).
        """
        start = time.process_time()
        if not (self.is_sources_set and self.is_servers_set):
            raise RuntimeError("set sources and servers before run()")

        inf = float("inf")
        t = 0.0
        next_arr = [self.sources[k].generate() for k in range(self.n_classes)]
        waiting: list[list[float]] = [[] for _ in range(self.n_classes)]  # arrival times (FIFO per class)
        busy_until = inf
        serving_class = -1  # class in service (-1 = idle)
        serving_arrived = 0.0

        served = 0
        warm = int(total_served * warmup_fraction)
        stats_on = False
        w_sum = [0.0] * self.n_classes
        v_sum = [0.0] * self.n_classes
        cnt = [0] * self.n_classes
        v_cnt = [0] * self.n_classes

        def start_service(k: int, arrived: float):
            nonlocal busy_until, serving_class, serving_arrived
            serving_class, serving_arrived = k, arrived
            busy_until = t + self.servers[k].generate()
            if stats_on:
                w_sum[k] += t - arrived
                cnt[k] += 1

        def pick_next():
            best_k, best_credit = -1, -inf
            for k in range(self.n_classes):
                if waiting[k]:
                    credit = self.rates[k] * (t - waiting[k][0])
                    if credit > best_credit:
                        best_k, best_credit = k, credit
            return best_k

        while served < total_served + warm:
            arr_min = min(next_arr)
            t = min(arr_min, busy_until)
            if busy_until <= arr_min:
                k = serving_class
                if stats_on:
                    v_sum[k] += t - serving_arrived
                    v_cnt[k] += 1
                served += 1
                if not stats_on and served >= warm:
                    stats_on = True
                    for j in range(self.n_classes):
                        w_sum[j] = v_sum[j] = 0.0
                        cnt[j] = v_cnt[j] = 0
                nxt = pick_next()
                if nxt >= 0:
                    start_service(nxt, waiting[nxt].pop(0))
                else:
                    busy_until = inf
                    serving_class = -1
                continue
            k = min(range(self.n_classes), key=lambda j: next_arr[j])
            if serving_class < 0:
                start_service(k, t)
            else:
                waiting[k].append(t)
            next_arr[k] = t + self.sources[k].generate()

        w = [[w_sum[k] / cnt[k] if cnt[k] else 0.0] for k in range(self.n_classes)]
        v = [[v_sum[k] / v_cnt[k] if v_cnt[k] else 0.0] for k in range(self.n_classes)]
        return MulticlassResults(w=w, v=v, duration=time.process_time() - start)
