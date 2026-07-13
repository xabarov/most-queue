"""
Tandem queueing network with finite buffers and blocking-after-service (BAS).

A line of M exponential single-server nodes; node i holds at most K_i jobs
(including the one in service). An external Poisson flow enters node 1 (an
arrival finding node 1 full is lost). A job finishing service at node i moves
to node i+1; if node i+1 is full, the job stays on node i's server and blocks
it until space frees downstream (blocking after service, BAS).

Approximate solution by the classic two-pass decomposition (Takahashi,
Miyahara & Hasegawa, 1980; Brandwajn & Jow, Operations Research 1988;
Dallery & Frein, Operations Research 1993; survey — Perros / Balsamo et al.,
"Analysis of Queueing Networks with Blocking", Kluwer 2001). Each node is
analyzed as an M/M/1/K queue:

- backward pass: the effective service time of node i is inflated by the
  probability of blocking at node i+1:
      1/mu_eff_i = 1/mu_i + Pb_{i+1} / mu_eff_{i+1},
  where Pb_{i+1} is the probability that node i+1 is full;
- forward pass: node i is solved as M/M/1/K_i with arrival rate equal to the
  throughput of node i-1, giving its full probability and throughput.

The two passes are iterated to a fixed point. Buffer -> infinity reduces to
the open Jackson tandem.
"""

import time

import numpy as np

from most_queue.structs import NetworkMeansResults
from most_queue.theory.networks.base_network_calc import BaseNetwork


def mm1k_probs(lam: float, mu: float, k: int) -> np.ndarray:
    """
    Stationary distribution of M/M/1/K (K = capacity including in service).
    """
    rho = lam / mu
    if abs(rho - 1.0) < 1e-12:
        return np.full(k + 1, 1.0 / (k + 1))
    p0 = (1.0 - rho) / (1.0 - rho ** (k + 1))
    return p0 * rho ** np.arange(k + 1)


class TandemBlockingCalc(BaseNetwork):
    """
    Two-pass decomposition for an exponential tandem with finite buffers
    and blocking after service.

    :param max_iter: fixed-point iteration limit.
    :param tol: convergence tolerance on throughput.
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-12):
        super().__init__()
        self.arrival_rate = None
        self.mu = None
        self.capacity = None
        self.max_iter = max_iter
        self.tol = tol
        self.blocking_probs = None  # P(node i+1 full) seen by node i completions
        self.loss_prob = None  # external arrivals lost at node 1
        self.throughput = None

    def set_sources(self, arrival_rate: float):  # pylint: disable=arguments-differ
        """
        :param arrival_rate: external Poisson rate into node 1.
        """
        self.arrival_rate = float(arrival_rate)
        self.is_sources_set = True

    def set_nodes(self, mu: list[float], capacity: list[int]):  # pylint: disable=arguments-differ
        """
        :param mu: exponential service rate of each node.
        :param capacity: K_i — max jobs at node i (queue + in service);
            None or math.inf marks an unlimited node.
        """
        self.mu = [float(x) for x in mu]
        self.capacity = [None if (k is None or k == float("inf")) else int(k) for k in capacity]
        for i, k in enumerate(self.capacity):
            if k is not None and k < 1:
                raise ValueError(f"Node {i}: capacity must be >= 1")
        self.is_nodes_set = True

    @staticmethod
    def _node_full_prob(lam: float, mu: float, k: int | None) -> float:
        if k is None:
            return 0.0
        return float(mm1k_probs(lam, mu, k)[-1])

    def run(self) -> NetworkMeansResults:
        """
        Iterate the backward (effective rates) and forward (flows) passes.
        """
        start = time.process_time()
        self._check_sources_and_nodes_is_set()

        m = len(self.mu)
        mu_eff = np.array(self.mu, dtype=float)
        lam_in = np.full(m, self.arrival_rate)
        x_prev = 0.0

        for _ in range(self.max_iter):
            # Backward pass: effective service rates under BAS blocking
            for i in range(m - 2, -1, -1):
                pb_next = self._node_full_prob(lam_in[i + 1], mu_eff[i + 1], self.capacity[i + 1])
                mu_eff[i] = 1.0 / (1.0 / self.mu[i] + pb_next / mu_eff[i + 1])

            # Forward pass: jobs are lost only at the entry (BAS conserves
            # flow at interior nodes), so every downstream node sees the
            # accepted throughput X.
            lam_in[0] = self.arrival_rate
            p_full_entry = self._node_full_prob(lam_in[0], mu_eff[0], self.capacity[0])
            throughput = lam_in[0] * (1.0 - p_full_entry)
            for i in range(1, m):
                lam_in[i] = throughput

            if abs(throughput - x_prev) < self.tol:
                break
            x_prev = throughput

        loads, mean_jobs, v_node, blocking = [], [], [], []
        for i in range(m):
            k = self.capacity[i]
            if k is None:
                rho = lam_in[i] / mu_eff[i]
                loads.append(rho)
                mean_jobs.append(rho / (1.0 - rho))
            else:
                probs = mm1k_probs(lam_in[i], mu_eff[i], k)
                loads.append(1.0 - probs[0])
                mean_jobs.append(float(np.dot(np.arange(k + 1), probs)))
            accepted = lam_in[i] * (1.0 - self._node_full_prob(lam_in[i], mu_eff[i], k))
            v_node.append(mean_jobs[i] / accepted if accepted > 0 else 0.0)
            if i + 1 < m:
                blocking.append(self._node_full_prob(lam_in[i + 1], mu_eff[i + 1], self.capacity[i + 1]))

        self.blocking_probs = [float(b) for b in blocking]
        self.loss_prob = self._node_full_prob(lam_in[0], mu_eff[0], self.capacity[0])
        self.throughput = float(x_prev)

        v_mean = float(sum(mean_jobs)) / self.throughput if self.throughput > 0 else 0.0

        self.results = NetworkMeansResults(
            v=[v_mean],
            intensities=[float(x) for x in lam_in],
            loads=[float(x) for x in loads],
            mean_jobs=[float(x) for x in mean_jobs],
            v_node=[float(x) for x in v_node],
            duration=time.process_time() - start,
        )
        return self.results
