"""
Time-varying open Markovian network: PSA (pointwise stationary approximation).

The external arrival rate is a function lambda(t); at every grid instant the
network is solved as a stationary Jackson network with the instantaneous rate
(PSA — Green & Kolesar). The approximation is accurate when the modulation is
slow relative to the relaxation time of the network (the same regime as the
single-node `TimeVaryingMMcCalc` from EPIC-016); for fast modulation PSA
overestimates the response to peaks.

The constant-rate special case reduces exactly to `JacksonNetworkCalc`.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from most_queue.theory.networks.jackson_network import JacksonNetworkCalc


@dataclass
class TimeVaryingNetworkResults:
    """PSA results on a time grid."""

    t: list = field(default_factory=list)
    v: list = field(default_factory=list)  # mean network sojourn time at each t
    mean_jobs_total: list = field(default_factory=list)  # sum_i L_i(t)
    loads: list = field(default_factory=list)  # per-node loads at each t
    duration: float = 0.0


class TimeVaryingNetworkCalc:
    """
    PSA over an open Jackson network with arrival rate lambda(t).
    """

    def __init__(self):
        self.lam_fn = None
        self.R = None
        self.mu = None
        self.n = None
        self.is_sources_set = False
        self.is_nodes_set = False
        self.results = None

    def set_sources(self, lam_fn, R):
        """
        :param lam_fn: callable t -> external arrival rate lambda(t).
        :param R: routing matrix, dim (m + 1 x m + 1) (same format as
            `OpenNetworkCalc`).
        """
        self.lam_fn = lam_fn
        self.R = np.asarray(R, dtype=float)
        self.is_sources_set = True

    def set_nodes(self, mu: list, n: list[int]):
        """
        :param mu: exponential service rate per channel at each node.
        :param n: number of channels at each node.
        """
        self.mu = [float(x) for x in mu]
        self.n = [int(x) for x in n]
        self.is_nodes_set = True

    def run(self, t_grid) -> TimeVaryingNetworkResults:
        """
        Solve the stationary Jackson network at every grid instant.

        Raises ValueError if the instantaneous load reaches 1 at some t (PSA
        requires pointwise stability).
        """
        start = time.process_time()
        if not (self.is_sources_set and self.is_nodes_set):
            raise ValueError("Sources and nodes must be set before run()")

        res = TimeVaryingNetworkResults()
        for t in t_grid:
            lam_t = self.lam_fn(t)
            calc = JacksonNetworkCalc()
            calc.set_sources(arrival_rate=lam_t, R=self.R)
            calc.set_nodes(mu=self.mu, n=self.n)
            try:
                snapshot = calc.run()
            except ValueError as exc:
                raise ValueError(f"PSA: network unstable at t={t} (lambda={lam_t}): {exc}") from exc
            res.t.append(float(t))
            res.v.append(snapshot.v[0])
            res.mean_jobs_total.append(float(sum(snapshot.mean_jobs)))
            res.loads.append(snapshot.loads)

        res.duration = time.process_time() - start
        self.results = res
        return res
