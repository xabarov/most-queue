"""
Open network with embedded fork-join stations.

A routed open network in which some nodes are ordinary M/M/n stations and
some are fork-join stations: an arriving job forks into k sub-tasks served
in parallel by k independent single-server exponential branches (each with
its own queue) and leaves the station when the last sub-task finishes.

Mean-value analysis by decomposition: the mean response time of each
fork-join station under Poisson arrivals is taken from the classic
approximations (Nelson & Tantawi, 1988; Varma & Makowski, 1994 — see
`ForkJoinMarkovianCalc`), ordinary stations are exact M/M/n, and the mean
network sojourn time follows from Little's law. Approximation quality for
fork-join response inside a network mirrors the standalone approximations
(a few percent); departure processes of fork-join stations are treated as
Poisson (the same assumption the plain decomposition makes). See also the
multiclass network treatment in IEEE TPDS 2013 (doi:10.1109/tpds.2013.70).

Node specification (`set_nodes`):
    {"kind": "queue", "mu": 1.0, "n": 2}          — M/M/n station
    {"kind": "fork_join", "mu": 1.0, "k": 3}      — fork-join with k branches
"""

import time

import numpy as np

from most_queue.structs import NetworkMeansResults
from most_queue.theory.fork_join.m_m_n import ForkJoinMarkovianCalc
from most_queue.theory.networks.base_network_calc import BaseNetwork
from most_queue.theory.networks.jackson_network import JacksonNetworkCalc
from most_queue.theory.networks.traffic import solve_traffic_equations


class OpenNetworkCalcForkJoin(BaseNetwork):
    """
    Mean-value decomposition for open networks with fork-join stations.

    :param fj_approx: approximation for fork-join response ("varma" or
        "nelson_tantawi", see `ForkJoinMarkovianCalc.run`).
    """

    def __init__(self, fj_approx: str = "varma"):
        super().__init__()
        self.arrival_rate = None
        self.R = None
        self.nodes = None
        self.fj_approx = fj_approx

    def set_sources(self, arrival_rate: float, R):  # pylint: disable=arguments-differ
        """
        :param arrival_rate: external arrival rate.
        :param R: routing matrix, dim (m + 1 x m + 1) (same format as
            `OpenNetworkCalc`).
        """
        self.arrival_rate = arrival_rate
        self.R = np.asarray(R, dtype=float)
        self.is_sources_set = True

    def set_nodes(self, nodes: list[dict]):  # pylint: disable=arguments-differ
        """
        :param nodes: per-node spec: {"kind": "queue", "mu": .., "n": ..} or
            {"kind": "fork_join", "mu": .., "k": ..}.
        """
        for i, node in enumerate(nodes):
            if node.get("kind") not in ("queue", "fork_join"):
                raise ValueError(f"Node {i}: kind must be 'queue' or 'fork_join'")
        self.nodes = nodes
        self.is_nodes_set = True

    def _node_mean_sojourn(self, node: dict, lam: float) -> float:
        if lam < 1e-12:
            return 0.0
        if node["kind"] == "queue":
            _, w = JacksonNetworkCalc._mmn_metrics(lam, node["mu"], node["n"])  # pylint: disable=protected-access
            return w
        fj = ForkJoinMarkovianCalc(n=node["k"])
        fj.set_sources(l=lam)
        fj.set_servers(mu=node["mu"])
        return fj.run(approx=self.fj_approx).v[0]

    def run(self) -> NetworkMeansResults:
        """
        Run the mean-value decomposition.
        """
        start = time.process_time()
        self._check_sources_and_nodes_is_set()

        lam = solve_traffic_equations(self.arrival_rate, self.R)

        v_node, mean_jobs, loads = [], [], []
        for i, node in enumerate(self.nodes):
            w = self._node_mean_sojourn(node, lam[i])
            v_node.append(w)
            mean_jobs.append(lam[i] * w)
            if node["kind"] == "queue":
                load = lam[i] / (node["mu"] * node["n"])
            else:
                # every branch of a fork-join station receives the full flow
                load = lam[i] / node["mu"]
            if load >= 1.0:
                raise ValueError(f"Node {i} is unstable: utilization {load:.3f} >= 1")
            loads.append(load)

        v_mean = sum(mean_jobs) / self.arrival_rate

        self.results = NetworkMeansResults(
            v=[float(v_mean)],
            intensities=[float(x) for x in lam],
            loads=[float(x) for x in loads],
            mean_jobs=[float(x) for x in mean_jobs],
            v_node=[float(x) for x in v_node],
            duration=time.process_time() - start,
        )
        return self.results
