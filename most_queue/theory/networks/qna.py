"""
QNA — Queueing Network Analyzer (Whitt, Bell System Technical Journal, 1983).

Two-moment parametric decomposition of an open network with general service
and non-Poisson internal flows. Unlike the plain decomposition of
`OpenNetworkCalc` (which treats every internal flow as Poisson), QNA
propagates the squared coefficient of variation (scv) of interarrival times
through the three network operations:

- departure:     c2_d = 1 + rho^2 (c2_s - 1)/sqrt(n) + (1 - rho^2)(c2_a - 1)
- splitting (p): c2   = p c2 + 1 - p
- superposition: hybrid of the asymptotic weighted average and the Poisson
                 limit with Whitt's traffic-intensity weight w.

Each node is then approximated as an independent GI/G/n queue with the
Kraemer & Langenbach-Belz correction for the mean wait:

  Wq = g * (c2_a + c2_s)/2 * Wq(M/M/n)

Only mean values are produced (v[0] — network mean sojourn time by Little's
law); per-node means are in `v_node` / `mean_jobs`.
"""

import math
import time

import numpy as np

from most_queue.structs import NetworkMeansResults
from most_queue.theory.networks.base_network_calc import BaseNetwork
from most_queue.theory.networks.traffic import solve_traffic_equations


class OpenNetworkCalcQNA(BaseNetwork):
    """
    QNA (Whitt) calculator for open networks with GI/G/n nodes.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-10):
        super().__init__()
        self.R = None
        self.arrival_rate = None
        self.arrival_cv2 = 1.0
        self.b = None
        self.n = None
        self.max_iter = max_iter
        self.tol = tol
        self.arrival_cv2_nodes = None  # c2_a at each node after the fixed point

    def set_sources(self, arrival_rate: float, R, arrival_cv2: float = 1.0):  # pylint: disable=arguments-differ
        """
        Set the external arrival flow and routing matrix.

        :param arrival_rate: external arrival rate.
        :param R: routing matrix, dim (m + 1 x m + 1) (same format as
            `OpenNetworkCalc`).
        :param arrival_cv2: squared coefficient of variation of external
            interarrival times (1 = Poisson).
        """
        self.arrival_rate = arrival_rate
        self.R = np.asarray(R, dtype=float)
        self.arrival_cv2 = arrival_cv2
        self.is_sources_set = True

    def set_nodes(self, b: list[list[float]], n: list[int]):  # pylint: disable=arguments-differ
        """
        Set the service time distribution moments and number of channels.

        :param b: raw moments [E[S], E[S^2], ...] of service time per node.
        :param n: number of channels per node.
        """
        self.b = b
        self.n = [int(x) for x in n]
        self.is_nodes_set = True

    @staticmethod
    def _wq_mmn(lam: float, mu: float, n: int) -> float:
        """
        Exact mean wait of M/M/n (Erlang C).
        """
        a = lam / mu
        rho = a / n
        p0_inv = sum(a**k / math.factorial(k) for k in range(n)) + a**n / (math.factorial(n) * (1.0 - rho))
        erlang_c = a**n / (math.factorial(n) * (1.0 - rho)) / p0_inv
        return erlang_c / (n * mu - lam)

    @staticmethod
    def _klb_correction(rho: float, c2a: float, c2s: float) -> float:
        """
        Kraemer & Langenbach-Belz correction factor g(rho, c2a, c2s).
        """
        if c2a >= 1.0:
            return 1.0
        return math.exp(-2.0 * (1.0 - rho) * (1.0 - c2a) ** 2 / (3.0 * rho * (c2a + c2s)))

    def run(self) -> NetworkMeansResults:
        """
        Run the QNA calculation.
        """
        start = time.process_time()
        self._check_sources_and_nodes_is_set()

        m = len(self.n)
        lam = solve_traffic_equations(self.arrival_rate, self.R)
        lam_ext = self.arrival_rate * self.R[0, :m]
        Q = self.R[1:, :m]  # node-to-node routing probabilities

        b1 = np.array([bi[0] for bi in self.b])
        c2s = np.array([(bi[1] - bi[0] ** 2) / bi[0] ** 2 for bi in self.b])
        rho = lam * b1 / np.array(self.n)
        if np.any(rho >= 1.0):
            raise ValueError(f"Network is unstable: node loads {rho}")

        # Fixed point on arrival scv per node
        c2a = np.ones(m)
        for _ in range(self.max_iter):
            c2a_prev = c2a.copy()
            c2d = 1.0 + rho**2 * (c2s - 1.0) / np.sqrt(np.array(self.n)) + (1.0 - rho**2) * (c2a - 1.0)
            for j in range(m):
                if lam[j] < 1e-12:
                    continue
                rates, scvs = [], []
                if lam_ext[j] > 1e-12:
                    p = self.R[0, j]
                    rates.append(lam_ext[j])
                    scvs.append(p * self.arrival_cv2 + 1.0 - p)
                for i in range(m):
                    if lam[i] * Q[i, j] > 1e-12:
                        p = Q[i, j]
                        rates.append(lam[i] * p)
                        scvs.append(p * c2d[i] + 1.0 - p)
                weights = np.array(rates) / lam[j]
                c2_asympt = float(np.dot(weights, scvs))
                v = 1.0 / float(np.sum(weights**2))
                w = 1.0 / (1.0 + 4.0 * (1.0 - rho[j]) ** 2 * (v - 1.0))
                c2a[j] = w * c2_asympt + (1.0 - w)
            if np.max(np.abs(c2a - c2a_prev)) < self.tol:
                break

        self.arrival_cv2_nodes = [float(x) for x in c2a]

        # GI/G/n nodes: KLB-corrected mean waits
        v_node, mean_jobs = [], []
        for i in range(m):
            mu_i = 1.0 / b1[i]
            wq = self._wq_mmn(lam[i], mu_i, self.n[i])
            wq *= (c2a[i] + c2s[i]) / 2.0
            if self.n[i] == 1:
                wq *= self._klb_correction(rho[i], c2a[i], c2s[i])
            w_i = wq + b1[i]
            v_node.append(w_i)
            mean_jobs.append(lam[i] * w_i)

        v_mean = sum(mean_jobs) / self.arrival_rate

        self.results = NetworkMeansResults(
            v=[float(v_mean)],
            intensities=[float(x) for x in lam],
            loads=[float(x) for x in rho],
            mean_jobs=[float(x) for x in mean_jobs],
            v_node=[float(x) for x in v_node],
            duration=time.process_time() - start,
        )
        return self.results
