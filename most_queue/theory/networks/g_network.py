"""
G-network (Gelenbe): exact product-form solution for an open network with
positive and negative customers (Gelenbe, Journal of Applied Probability,
1991).

M/M/1 nodes. A job completing service at node i goes to node j as a positive
customer with probability P_plus[i, j], as a negative customer (signal) with
probability P_minus[i, j], or leaves the network with the complementary
probability. A negative customer arriving at a non-empty node removes one
customer and vanishes; at an empty node it just vanishes. External positive
and negative arrivals are Poisson with per-node rates.

The stationary distribution has product form P(k) = prod_i (1 - rho_i) *
rho_i^k_i, where the effective loads rho_i solve the nonlinear traffic
equations

    lambda_plus_i  = ext_plus_i  + sum_j rho_j mu_j P_plus[j, i]
    lambda_minus_i = ext_minus_i + sum_j rho_j mu_j P_minus[j, i]
    rho_i = lambda_plus_i / (mu_i + lambda_minus_i)

solved here by fixed-point iteration. With no negative customers the model
reduces to an open Jackson network.

v[0] is the mean time in the network per external positive arrival (Little's
law over the whole network; the sojourn of jobs later destroyed by signals is
counted up to the destruction instant).
"""

import time

import numpy as np

from most_queue.structs import NetworkMeansResults
from most_queue.theory.networks.base_network_calc import BaseNetwork


class GNetworkCalc(BaseNetwork):
    """
    Exact product-form calculator for Gelenbe G-networks (M/M/1 nodes).

    :param max_iter: fixed-point iteration limit for the traffic equations.
    :param tol: fixed-point convergence tolerance.
    """

    def __init__(self, max_iter: int = 10000, tol: float = 1e-12):
        super().__init__()
        self.positive_rates = None
        self.negative_rates = None
        self.P_plus = None
        self.P_minus = None
        self.mu = None
        self.max_iter = max_iter
        self.tol = tol
        self.loads_solution = None  # rho_i after the fixed point

    def set_sources(
        self,
        positive_rates: list[float],
        P_plus,
        P_minus=None,
        negative_rates: list[float] | None = None,
    ):  # pylint: disable=arguments-differ
        """
        Set external flows and routing.

        :param positive_rates: external Poisson rates of positive customers
            per node.
        :param P_plus: matrix (m x m) — probability that a job completing
            service at node i moves to node j as a positive customer.
        :param P_minus: matrix (m x m) — probability that a job completing
            service at node i moves to node j as a negative customer (signal);
            None means no signal routing.
        :param negative_rates: external Poisson rates of negative customers
            per node (None — no external negatives).

        The exit probability of node i is 1 - sum_j (P_plus[i,j] + P_minus[i,j]).
        """
        self.positive_rates = np.asarray(positive_rates, dtype=float)
        m = len(self.positive_rates)
        self.P_plus = np.asarray(P_plus, dtype=float)
        self.P_minus = np.zeros((m, m)) if P_minus is None else np.asarray(P_minus, dtype=float)
        self.negative_rates = np.zeros(m) if negative_rates is None else np.asarray(negative_rates, dtype=float)
        depart = self.P_plus.sum(axis=1) + self.P_minus.sum(axis=1)
        if np.any(depart > 1.0 + 1e-9):
            raise ValueError("Rows of P_plus + P_minus must sum to <= 1")
        self.is_sources_set = True

    def set_nodes(self, mu: list[float]):  # pylint: disable=arguments-differ
        """
        :param mu: exponential service rate of each (single-channel) node.
        """
        self.mu = np.asarray(mu, dtype=float)
        self.is_nodes_set = True

    def run(self) -> NetworkMeansResults:
        """
        Solve the nonlinear traffic equations and evaluate the product form.
        """
        start = time.process_time()
        self._check_sources_and_nodes_is_set()

        m = len(self.mu)
        rho = np.zeros(m)
        lam_plus = self.positive_rates.copy()
        lam_minus = self.negative_rates.copy()

        for _ in range(self.max_iter):
            rho_prev = rho.copy()
            served_flow = rho * self.mu  # throughput of each node
            lam_plus = self.positive_rates + served_flow @ self.P_plus
            lam_minus = self.negative_rates + served_flow @ self.P_minus
            rho = lam_plus / (self.mu + lam_minus)
            if np.max(np.abs(rho - rho_prev)) < self.tol:
                break

        if np.any(rho >= 1.0):
            raise ValueError(f"G-network is unstable: loads {rho}")

        self.loads_solution = [float(x) for x in rho]
        mean_jobs = rho / (1.0 - rho)
        v_node = [float(mean_jobs[i] / lam_plus[i]) if lam_plus[i] > 1e-12 else 0.0 for i in range(m)]
        total_external = float(self.positive_rates.sum())
        v_mean = float(mean_jobs.sum()) / total_external if total_external > 0 else 0.0

        self.results = NetworkMeansResults(
            v=[v_mean],
            intensities=[float(x) for x in lam_plus],
            loads=[float(x) for x in rho],
            mean_jobs=[float(x) for x in mean_jobs],
            v_node=v_node,
            negative_intensities=[float(x) for x in lam_minus],
            duration=time.process_time() - start,
        )
        return self.results
