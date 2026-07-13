"""
Discrete-event simulator for a closed queueing network: a fixed population of
N jobs circulates over M multi-channel FCFS nodes (or infinite-server delay
nodes) according to a routing matrix. Validates the MVA / convolution
calculators (`ClosedNetworkCalc`).
"""

import math
import time

import numpy as np

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import ClosedNetworkResults


class ClosedNetworkSim(BaseSimulationCore):
    """
    Closed network simulator.

    :param seed: RNG seed.
    """

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)
        self.R = None
        self.N = None
        self.n = None
        self.servers = None
        self.n_nodes = 0
        self.is_sources_set = False
        self.is_nodes_set = False

    def set_sources(self, R: np.ndarray, N: int):
        """
        :param R: routing matrix (m x m), rows sum to 1 (closed network).
        :param N: population size.
        """
        self.R = np.asarray(R, dtype=float)
        self.N = int(N)
        self.is_sources_set = True

    def set_nodes(self, serv_params: list[dict], n: list):
        """
        :param serv_params: per-node service distribution, [{"type": kendall, "params": ...}].
        :param n: channels per node; None (or math.inf) marks an infinite-server (delay) node.
        """
        self.servers = [create_distribution(p["params"], p["type"], self.generator) for p in serv_params]
        self.n = [None if (x is None or x == math.inf) else int(x) for x in n]
        self.n_nodes = len(self.n)
        self.is_nodes_set = True

    def _choose_next(self, node: int) -> int:
        return int(self.generator.choice(self.n_nodes, p=self.R[node]))

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> ClosedNetworkResults:
        """
        Run until `total_served` service completions (after warmup).

        Returns throughput of node 0, mean jobs, per-visit sojourn and loads.
        """
        start = time.process_time()
        if not (self.is_sources_set and self.is_nodes_set):
            raise RuntimeError("set sources and nodes before run()")

        m = self.n_nodes
        t = 0.0
        # in_service[i] — list of (completion_time, arrival_time_at_node)
        in_service = [[] for _ in range(m)]
        queue = [[] for _ in range(m)]  # arrival times of waiting jobs (FCFS)
        jobs_at = [0] * m

        # Warmup boundary by served count; statistics collected after it
        warm = int(total_served * warmup_fraction)
        served = 0
        stats_on = False
        t0 = 0.0
        area_l = [0.0] * m  # integral of jobs_at
        busy_area = [0.0] * m  # integral of busy channels
        completions = [0] * m
        sojourn_sum = [0.0] * m
        sojourn_cnt = [0] * m

        def start_service(i: int, arrival: float):
            comp = t + self.servers[i].generate()
            in_service[i].append((comp, arrival))

        def on_arrival(i: int, when: float):
            jobs_at[i] += 1
            if self.n[i] is None or len(in_service[i]) < self.n[i]:
                start_service(i, when)
            else:
                queue[i].append(when)

        # All jobs start at node 0
        for _ in range(self.N):
            on_arrival(0, 0.0)

        while served < total_served + warm:
            # Next completion over all nodes
            node, k_min, t_next = -1, -1, float("inf")
            for i in range(m):
                for k, (comp, _) in enumerate(in_service[i]):
                    if comp < t_next:
                        node, k_min, t_next = i, k, comp

            if stats_on:
                dt = t_next - t
                for i in range(m):
                    area_l[i] += jobs_at[i] * dt
                    busy_area[i] += len(in_service[i]) * dt
            t = t_next

            _, arrived_at = in_service[node].pop(k_min)
            jobs_at[node] -= 1
            served += 1
            if stats_on:
                completions[node] += 1
                sojourn_sum[node] += t - arrived_at
                sojourn_cnt[node] += 1
            if queue[node]:
                start_service(node, queue[node].pop(0))

            on_arrival(self._choose_next(node), t)

            if not stats_on and served >= warm:
                stats_on = True
                t0 = t

        elapsed = t - t0
        intensities = [completions[i] / elapsed for i in range(m)]
        x = intensities[0]
        mean_jobs = [area_l[i] / elapsed for i in range(m)]
        v_node = [sojourn_sum[i] / sojourn_cnt[i] if sojourn_cnt[i] else 0.0 for i in range(m)]
        loads = [0.0 if self.n[i] is None else busy_area[i] / elapsed / self.n[i] for i in range(m)]

        return ClosedNetworkResults(
            v=[self.N / x if x > 0 else 0.0],
            intensities=intensities,
            loads=loads,
            throughput=x,
            mean_jobs=mean_jobs,
            v_node=v_node,
            served=served,
            duration=time.process_time() - start,
        )
