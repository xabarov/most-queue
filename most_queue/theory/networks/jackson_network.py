"""
Open Jackson network: exact product-form solution (Jackson, 1957/1963).

Poisson external arrivals, exponential M/M/n nodes, Markovian routing. The
stationary distribution factorizes over nodes, each node behaving as an
independent M/M/n queue fed by the flow-balance arrival rate. Mean
performance measures are exact; the mean network sojourn time follows from
Little's law, E[T] = sum_i L_i / Lambda. Higher sojourn moments are not
available in closed form (overtaking), so v contains only the mean.

Serves as an exact baseline for the approximate decomposition of
`OpenNetworkCalc` on Markovian networks.
"""

import math
import time

import numpy as np

from most_queue.structs import NetworkMeansResults
from most_queue.theory.networks.base_network_calc import BaseNetwork
from most_queue.theory.networks.traffic import solve_traffic_equations


class JacksonNetworkCalc(BaseNetwork):
    """
    Exact product-form calculator for open Jackson networks (M/M/n nodes).
    """

    def __init__(self):
        super().__init__()
        self.R = None
        self.arrival_rate = None
        self.mu = None  # service rate per channel at each node
        self.n = None

    def set_sources(self, arrival_rate: float, R):  # pylint: disable=arguments-differ
        """
        Set the arrival rate and routing matrix.

        :param arrival_rate: external arrival rate of customers.
        :param R: routing matrix, dim (m + 1 x m + 1), where m is the number
            of nodes (same format as `OpenNetworkCalc`): row 0 — transitions
            from the source, last column — transitions out of the system.
        """
        self.arrival_rate = arrival_rate
        self.R = np.asarray(R, dtype=float)
        self.is_sources_set = True

    def set_nodes(self, mu: list, n: list[int]):  # pylint: disable=arguments-differ
        """
        Set exponential service rates and number of channels for each node.

        :param mu: service rate per channel at each node.
        :param n: number of channels at each node.
        """
        self.mu = [float(x) for x in mu]
        self.n = [int(x) for x in n]
        self.is_nodes_set = True

    def solve_intensities(self) -> list[float]:
        """
        Solve the flow balance equations (available as soon as sources are set).
        """
        if not self.is_sources_set:
            raise ValueError("Sources are not set. Please use set_sources() method.")
        return [float(x) for x in solve_traffic_equations(self.arrival_rate, self.R)]

    @staticmethod
    def _mmn_metrics(lam: float, mu: float, n: int) -> tuple[float, float]:
        """
        Exact M/M/n mean metrics: (L — mean jobs in system, W — mean sojourn).
        """
        if lam < 1e-12:
            return 0.0, 0.0
        a = lam / mu  # offered load
        rho = a / n
        if rho >= 1.0:
            raise ValueError(f"Node is unstable: utilization {rho:.3f} >= 1")
        p0_inv = sum(a**k / math.factorial(k) for k in range(n)) + a**n / (math.factorial(n) * (1.0 - rho))
        p0 = 1.0 / p0_inv
        l_queue = p0 * a**n * rho / (math.factorial(n) * (1.0 - rho) ** 2)
        w = l_queue / lam + 1.0 / mu
        return lam * w, w

    def run(self) -> NetworkMeansResults:
        """
        Run the exact product-form calculation.
        """
        start = time.process_time()
        self._check_sources_and_nodes_is_set()

        nodes = len(self.n)
        intensities = solve_traffic_equations(self.arrival_rate, self.R)

        mean_jobs = []
        v_node = []
        loads = []
        for i in range(nodes):
            big_l, w = self._mmn_metrics(intensities[i], self.mu[i], self.n[i])
            mean_jobs.append(big_l)
            v_node.append(w)
            loads.append(intensities[i] / (self.n[i] * self.mu[i]))

        # Little's law over the whole network: exact mean sojourn time
        v_mean = sum(mean_jobs) / self.arrival_rate

        self.results = NetworkMeansResults(
            v=[float(v_mean)],
            intensities=[float(x) for x in intensities],
            loads=[float(x) for x in loads],
            mean_jobs=[float(x) for x in mean_jobs],
            v_node=[float(x) for x in v_node],
            duration=time.process_time() - start,
        )
        return self.results
