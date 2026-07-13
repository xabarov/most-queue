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

`GNetworkMulticlassCalc` extends the model to multiple classes (Gelenbe,
Theoretical Computer Science, 1996): processor-sharing nodes with
class-dependent rates; a class-r signal arriving at a node removes a class-r
positive customer with probability equal to the class's service share
k_ir / k_i (and vanishes otherwise). The stationary law is the product of
per-node multinomial-geometric factors with q_ir = lam_plus_ir /
(mu_ir + lam_minus_ir); mean class-r jobs at node i: L_ir = q_ir / (1 - q_i).
The kill semantics was pinned down numerically: this variant satisfies the
exact global balance to truncation error, while "always kill if present"
does not.
"""

import time

import numpy as np

from most_queue.structs import BCMPNetworkResults, NetworkMeansResults
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


class GNetworkMulticlassCalc:
    """
    Multiclass G-network with processor-sharing nodes (Gelenbe, 1996).

    Class-r positive customers are served at node i with PS rate
    mu[i][r] * k_ir / k_i; a class-r signal removes a class-r customer with
    probability k_ir / k_i. Exact product form; mean values per class.

    :param max_iter: fixed-point iteration limit for the traffic equations.
    :param tol: fixed-point convergence tolerance.
    """

    def __init__(self, max_iter: int = 20000, tol: float = 1e-14):
        self.positive_rates = None  # [i][r]
        self.negative_rates = None  # [i][r]
        self.P_plus = None  # per class, m x m
        self.P_minus = None  # per class, m x m
        self.mu = None  # [i][r]
        self.max_iter = max_iter
        self.tol = tol
        self.is_sources_set = False
        self.is_nodes_set = False
        self.results = None
        self.q = None  # q_ir after the fixed point

    def set_sources(
        self,
        positive_rates,
        P_plus: list,
        P_minus: list | None = None,
        negative_rates=None,
    ):
        """
        :param positive_rates: external positive rates, [i][r].
        :param P_plus: per-class routing matrices (m x m) for positive moves.
        :param P_minus: per-class routing matrices (m x m) for signals (None — no signals).
        :param negative_rates: external negative rates, [i][r] (None — no external negatives).
        """
        self.positive_rates = np.asarray(positive_rates, dtype=float)
        m, n_classes = self.positive_rates.shape
        self.P_plus = [np.asarray(p, dtype=float) for p in P_plus]
        if len(self.P_plus) != n_classes:
            raise ValueError("Need one P_plus matrix per class")
        if P_minus is None:
            self.P_minus = [np.zeros((m, m)) for _ in range(n_classes)]
        else:
            self.P_minus = [np.asarray(p, dtype=float) for p in P_minus]
        self.negative_rates = (
            np.zeros((m, n_classes)) if negative_rates is None else np.asarray(negative_rates, dtype=float)
        )
        for r in range(n_classes):
            depart = self.P_plus[r].sum(axis=1) + self.P_minus[r].sum(axis=1)
            if np.any(depart > 1.0 + 1e-9):
                raise ValueError(f"Class {r}: rows of P_plus + P_minus must sum to <= 1")
        self.is_sources_set = True

    def set_nodes(self, mu):
        """
        :param mu: PS service rates, mu[i][r] — node i, class r.
        """
        self.mu = np.asarray(mu, dtype=float)
        self.is_nodes_set = True

    def run(self) -> BCMPNetworkResults:
        """
        Solve the per-class nonlinear traffic equations and evaluate the
        product form.
        """
        start = time.process_time()
        if not (self.is_sources_set and self.is_nodes_set):
            raise ValueError("Sources and nodes must be set before run()")

        m, n_classes = self.positive_rates.shape
        q = np.zeros((m, n_classes))
        lam_plus = self.positive_rates.copy()
        lam_minus = self.negative_rates.copy()

        for _ in range(self.max_iter):
            q_prev = q.copy()
            lam_plus = self.positive_rates.copy()
            lam_minus = self.negative_rates.copy()
            for r in range(n_classes):
                flow_r = q[:, r] * self.mu[:, r]  # class-r service completions
                lam_plus[:, r] += flow_r @ self.P_plus[r]
                lam_minus[:, r] += flow_r @ self.P_minus[r]
            q = lam_plus / (self.mu + lam_minus)
            if np.max(np.abs(q - q_prev)) < self.tol:
                break

        q_total = q.sum(axis=1)
        if np.any(q_total >= 1.0):
            raise ValueError(f"G-network is unstable: node loads {q_total}")

        self.q = q
        mean_jobs = q / (1.0 - q_total)[:, None]  # L_ir

        ext_per_class = self.positive_rates.sum(axis=0)
        v = []
        for r in range(n_classes):
            if ext_per_class[r] > 1e-12:
                v.append([float(mean_jobs[:, r].sum() / ext_per_class[r])])
            else:
                v.append([0.0])

        self.results = BCMPNetworkResults(
            v=v,
            intensities=[[float(lam_plus[i, r]) for i in range(m)] for r in range(n_classes)],
            loads=[float(x) for x in q_total],
            mean_jobs=[[float(mean_jobs[i, r]) for i in range(m)] for r in range(n_classes)],
            duration=time.process_time() - start,
        )
        return self.results
