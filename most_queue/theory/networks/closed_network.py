"""
Closed queueing networks (Gordon-Newell): exact MVA, Buzen convolution and
Schweitzer approximate MVA.

A fixed population of N jobs circulates over M nodes according to a routing
matrix R (rows sum to 1, no external arrivals). Visit ratios e_i are the
solution of e = e R normalized so that e_ref = 1 for the reference node.

Methods
-------
- "mva": exact Mean Value Analysis (Reiser & Lavenberg, JACM 1980), including
  exact treatment of multi-server stations via marginal queue-length
  probabilities (Bolch et al., Queueing Networks and Markov Chains, §8.2) and
  infinite-server (delay) stations.
- "convolution": Buzen's convolution algorithm for the normalization constant
  G(N) (Buzen, CACM 1973), with load-dependent factors for multi-server and
  delay stations. Cross-checks MVA to machine precision.
- "schweitzer": Schweitzer-Bard approximate MVA (fixed point on
  L_i(N-1) ~= L_i(N) * (N-1)/N) for large populations; multi-server stations
  are treated by the Seidmann approximation (single server of rate m*mu plus
  a delay of s*(m-1)/m).

Service is exponential (only the mean matters for the product-form solution
with FCFS exponential nodes; PS/LCFS-PR nodes are insensitive, so the same
formulas apply to their means with general service).
"""

import math
import time

import numpy as np

from most_queue.structs import ClosedNetworkResults
from most_queue.theory.networks.base_network_calc import BaseNetwork


class ClosedNetworkCalc(BaseNetwork):
    """
    Closed queueing network (Gordon-Newell) calculator.

    :param method: "mva" (exact), "convolution" (Buzen) or "schweitzer"
        (approximate MVA).
    """

    def __init__(self, method: str = "mva"):
        super().__init__()
        self.method = method.lower()
        if self.method not in ("mva", "convolution", "schweitzer"):
            raise ValueError("method must be 'mva', 'convolution' or 'schweitzer'")
        self.R = None
        self.N = None
        self.b = None  # mean service time per node
        self.n = None  # channels per node (None = infinite-server / delay node)
        self.e = None  # visit ratios

    def set_sources(self, R: np.ndarray, N: int):  # pylint: disable=arguments-differ
        """
        Set the routing matrix and population size.

        :param R: routing matrix, dim (m x m) — R[i, j] is the probability of
            going to node j after service completion at node i; rows sum to 1
            (the network is closed, there is no external source).
        :param N: number of jobs circulating in the network.
        """
        R = np.asarray(R, dtype=float)
        if R.shape[0] != R.shape[1]:
            raise ValueError("Routing matrix of a closed network must be square (m x m)")
        if not np.allclose(R.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Rows of the routing matrix of a closed network must sum to 1")
        if N < 1:
            raise ValueError("Population N must be >= 1")
        self.R = R
        self.N = int(N)
        self.is_sources_set = True

    def set_nodes(self, b: list, n: list):  # pylint: disable=arguments-differ
        """
        Set the mean service times and number of channels for each node.

        :param b: mean service time per node; each element is either a float
            or a list of raw moments (only the first moment is used).
        :param n: number of channels per node; None (or math.inf) marks an
            infinite-server (delay) node.
        """
        self.b = [x[0] if hasattr(x, "__len__") else float(x) for x in b]
        self.n = [None if (x is None or x == math.inf) else int(x) for x in n]
        for i, ni in enumerate(self.n):
            if ni is not None and ni < 1:
                raise ValueError(f"Node {i}: number of channels must be >= 1 or None for a delay node")
        self.is_nodes_set = True

    def visit_ratios(self) -> np.ndarray:
        """
        Solve e = e R with e[0] = 1 (node 0 is the reference node).
        """
        m = self.R.shape[0]
        A = self.R.T - np.eye(m)
        # Replace the first equation with the normalization e[0] = 1
        A[0, :] = 0.0
        A[0, 0] = 1.0
        rhs = np.zeros(m)
        rhs[0] = 1.0
        return np.linalg.solve(A, rhs)

    def run(self) -> ClosedNetworkResults:
        """
        Run the calculation.
        """
        start = time.process_time()
        self._check_sources_and_nodes_is_set()
        if len(self.b) != self.R.shape[0]:
            raise ValueError("Number of nodes in b and R must match")

        self.e = self.visit_ratios()

        if self.method == "mva":
            x, big_l, w = self._run_mva_exact()
        elif self.method == "schweitzer":
            x, big_l, w = self._run_schweitzer()
        else:
            x, big_l, w = self._run_convolution()

        intensities = [float(x * self.e[i]) for i in range(len(self.b))]
        loads = []
        for i, ni in enumerate(self.n):
            if ni is None:
                loads.append(0.0)  # delay node has no queueing, utilization per server is not defined
            else:
                loads.append(float(intensities[i] * self.b[i] / ni))

        self.results = ClosedNetworkResults(
            v=[float(self.N / x)],
            intensities=intensities,
            loads=loads,
            throughput=float(x),
            mean_jobs=[float(l) for l in big_l],
            v_node=[float(wi) for wi in w],
            duration=time.process_time() - start,
        )
        return self.results

    def _run_mva_exact(self):
        """
        Exact MVA (Reiser-Lavenberg 1980); multi-server stations via marginal
        probabilities (Bolch et al., §8.2.1).
        """
        m = len(self.b)
        big_l = np.zeros(m)
        w = np.zeros(m)
        x = 0.0

        # Marginal queue-length probabilities pi[i][j] = P(j jobs at node i)
        # for multi-server nodes, carried over from population k-1.
        pi = {}
        for i, ni in enumerate(self.n):
            if ni is not None and ni > 1:
                pi[i] = np.zeros(ni)
                pi[i][0] = 1.0  # empty network

        for k in range(1, self.N + 1):
            for i, ni in enumerate(self.n):
                if ni is None:
                    w[i] = self.b[i]
                elif ni == 1:
                    w[i] = self.b[i] * (1.0 + big_l[i])
                else:
                    correction = sum((ni - j - 1) * pi[i][j] for j in range(ni - 1))
                    w[i] = self.b[i] / ni * (1.0 + big_l[i] + correction)

            x = k / float(np.dot(self.e, w))
            big_l = x * self.e * w

            for i, ni in enumerate(self.n):
                if ni is not None and ni > 1:
                    lam_i = x * self.e[i]
                    new_pi = np.zeros(ni)
                    for j in range(1, ni):
                        new_pi[j] = lam_i * self.b[i] / j * pi[i][j - 1]
                    busy_share = lam_i * self.b[i]
                    new_pi[0] = 1.0 - (busy_share + sum((ni - j) * new_pi[j] for j in range(1, ni))) / ni
                    pi[i] = new_pi

        return x, big_l, w

    def _run_schweitzer(self, tol: float = 1e-10, max_iter: int = 10000):
        """
        Schweitzer-Bard approximate MVA: L_i(N-1) ~= L_i(N) * (N-1)/N.
        Multi-server stations via the Seidmann approximation.
        """
        m = len(self.b)
        n_pop = self.N
        big_l = np.full(m, n_pop / m)
        w = np.zeros(m)
        x = 0.0

        for _ in range(max_iter):
            l_prev = big_l.copy()
            l_reduced = big_l * (n_pop - 1) / n_pop
            for i, ni in enumerate(self.n):
                if ni is None:
                    w[i] = self.b[i]
                elif ni == 1:
                    w[i] = self.b[i] * (1.0 + l_reduced[i])
                else:
                    # Seidmann: single server of rate ni/b + delay b*(ni-1)/ni
                    w[i] = self.b[i] / ni * (1.0 + l_reduced[i]) + self.b[i] * (ni - 1) / ni
            x = n_pop / float(np.dot(self.e, w))
            big_l = x * self.e * w
            if np.max(np.abs(big_l - l_prev)) < tol:
                break

        return x, big_l, w

    def _load_factors(self, i: int) -> np.ndarray:
        """
        f_i(j), j = 0..N — load-dependent product factors of node i for the
        convolution algorithm: f_i(j) = y_i^j / prod_{l=1..j} alpha_i(l),
        where alpha_i(l) = min(l, n_i) (and alpha = l for a delay node).
        """
        y = self.e[i] * self.b[i]
        f = np.zeros(self.N + 1)
        f[0] = 1.0
        ni = self.n[i]
        for j in range(1, self.N + 1):
            alpha = j if ni is None else min(j, ni)
            f[j] = f[j - 1] * y / alpha
        return f

    def _run_convolution(self):
        """
        Buzen's convolution algorithm. G = f_1 * f_2 * ... * f_m (truncated
        convolution up to N); X(N) = G(N-1)/G(N); marginal distributions of
        node i via the complement convolution G^{(i)}.
        """
        m = len(self.b)
        n_pop = self.N

        factors = [self._load_factors(i) for i in range(m)]

        def convolve(a, b):
            g = np.zeros(n_pop + 1)
            for k in range(n_pop + 1):
                g[k] = sum(a[j] * b[k - j] for j in range(k + 1))
            return g

        g_full = factors[0]
        for i in range(1, m):
            g_full = convolve(g_full, factors[i])

        x = g_full[n_pop - 1] / g_full[n_pop]

        big_l = np.zeros(m)
        w = np.zeros(m)
        for i in range(m):
            # Complement convolution: all nodes except i
            g_c = None
            for j in range(m):
                if j == i:
                    continue
                g_c = factors[j] if g_c is None else convolve(g_c, factors[j])
            if g_c is None:  # single-node network
                g_c = np.zeros(n_pop + 1)
                g_c[0] = 1.0
            # Marginal distribution of node i and mean jobs
            p_i = np.array([factors[i][j] * g_c[n_pop - j] / g_full[n_pop] for j in range(n_pop + 1)])
            big_l[i] = float(np.dot(np.arange(n_pop + 1), p_i))
            lam_i = x * self.e[i]
            w[i] = big_l[i] / lam_i if lam_i > 0 else 0.0

        return x, big_l, w
