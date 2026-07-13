"""
Discrete-event simulator for an open network with embedded fork-join
stations: ordinary multi-channel FCFS nodes plus fork-join nodes where a job
forks into k parallel sub-tasks (one independent single-server branch each)
and proceeds only when the last sub-task completes. Validates
`OpenNetworkCalcForkJoin`.
"""

import time

import numpy as np

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.structs import NetworkMeansResults


class ForkJoinNetworkSim(BaseSimulationCore):
    """
    Open network with fork-join stations.

    Node spec (`set_nodes`): {"kind": "queue", "serv": {"type": "M", "params": mu}, "n": 2}
    or {"kind": "fork_join", "serv": {...}, "k": 3}.

    :param seed: RNG seed.
    """

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)
        self.arrival_rate = None
        self.source = None
        self.R = None
        self.nodes = None
        self.m = 0
        self.is_sources_set = False
        self.is_nodes_set = False

    def set_sources(self, arrival_rate: float, R):
        """
        :param arrival_rate: external Poisson rate.
        :param R: routing matrix, dim (m + 1 x m + 1) (same format as
            `NetworkSimulator`).
        """
        self.arrival_rate = arrival_rate
        self.source = create_distribution(arrival_rate, "M", self.generator)
        self.R = np.asarray(R, dtype=float)
        self.is_sources_set = True

    def set_nodes(self, nodes: list[dict]):
        """
        :param nodes: per-node spec (see class docstring); service
            distributions are created per node (fork-join branches share one
            distribution object — i.i.d. draws).
        """
        self.nodes = []
        for spec in nodes:
            node = dict(spec)
            node["dist"] = create_distribution(spec["serv"]["params"], spec["serv"]["type"], self.generator)
            self.nodes.append(node)
        self.m = len(self.nodes)
        self.is_nodes_set = True

    def _route_from(self, row: int) -> int:
        return int(self.generator.choice(self.m + 1, p=self.R[row, : self.m + 1]))

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> NetworkMeansResults:
        """
        Run until `total_served` jobs leave the network (after warmup).
        """
        start = time.process_time()
        if not (self.is_sources_set and self.is_nodes_set):
            raise RuntimeError("set sources and nodes before run()")

        t = 0.0
        next_arrival = self.source.generate()
        next_job_id = 0
        entry_time = {}  # job id -> network entry time

        # Per node state
        queues = []  # queue kind: list of job ids; fork_join kind: list per branch
        busy = []  # queue kind: list of (comp, job); fork_join: per branch (comp, job) | None
        remaining = {}  # fork-join: job id -> outstanding sub-tasks at its current node

        for node in self.nodes:
            if node["kind"] == "queue":
                queues.append([])
                busy.append([])
            else:
                queues.append([[] for _ in range(node["k"])])
                busy.append([None] * node["k"])

        served = 0
        warm = int(total_served * warmup_fraction)
        stats_on = False
        v_sum = 0.0
        v_cnt = 0

        def node_accept(i: int, job: int):
            node = self.nodes[i]
            if node["kind"] == "queue":
                if len(busy[i]) < node["n"]:
                    busy[i].append((t + node["dist"].generate(), job))
                else:
                    queues[i].append(job)
            else:
                remaining[job] = node["k"]
                for b in range(node["k"]):
                    if busy[i][b] is None:
                        busy[i][b] = (t + node["dist"].generate(), job)
                    else:
                        queues[i][b].append(job)

        def depart(i: int, job: int):
            nonlocal served, stats_on, v_sum, v_cnt
            nxt = self._route_from(i + 1)
            if nxt == self.m:
                served += 1
                if stats_on:
                    v_sum += t - entry_time[job]
                    v_cnt += 1
                elif served >= warm:
                    stats_on = True
                del entry_time[job]
            else:
                node_accept(nxt, job)

        while served < total_served + warm:
            # Earliest event: external arrival or some completion
            t_next, which = next_arrival, ("arrival", 0, 0)
            for i, node in enumerate(self.nodes):
                if node["kind"] == "queue":
                    for c, (comp, _) in enumerate(busy[i]):
                        if comp < t_next:
                            t_next, which = comp, ("queue", i, c)
                else:
                    for b, slot in enumerate(busy[i]):
                        if slot is not None and slot[0] < t_next:
                            t_next, which = slot[0], ("fj", i, b)
            t = t_next

            kind, i, c = which
            if kind == "arrival":
                job = next_job_id
                next_job_id += 1
                entry_time[job] = t
                node_accept(self._route_from(0), job)
                next_arrival = t + self.source.generate()
            elif kind == "queue":
                _, job = busy[i].pop(c)
                if queues[i]:
                    busy[i].append((t + self.nodes[i]["dist"].generate(), queues[i].pop(0)))
                depart(i, job)
            else:  # fork-join branch completion
                _, job = busy[i][c]
                busy[i][c] = None
                if queues[i][c]:
                    busy[i][c] = (t + self.nodes[i]["dist"].generate(), queues[i][c].pop(0))
                remaining[job] -= 1
                if remaining[job] == 0:
                    del remaining[job]
                    depart(i, job)

        return NetworkMeansResults(
            v=[v_sum / v_cnt if v_cnt else 0.0],
            served=served,
            duration=time.process_time() - start,
        )
