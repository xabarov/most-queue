"""
Calculates queueing network.
"""

import time

import numpy as np

from most_queue.random.distributions import GammaDistribution
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.theory.networks.base_network_calc import BaseNetwork, NetworkResults
from most_queue.theory.utils.diff5dots import diff5dots
from most_queue.theory.utils.transforms import lst_gamma


class OpenNetworkCalc(BaseNetwork):
    """
    Calculates queueing network.
    """

    def __init__(self, is_markovian=False):
        """
        Initializes the OpenNetworkCalc class
        :param is_markovian: if True the nodes service distributions are Markovian (exponential)
        """
        super().__init__()

        self.R = None
        self.b = None
        self.n = None
        self.arrival_rate = None
        self.is_markovian = is_markovian

    def set_sources(self, arrival_rate: float, R: np.matrix):  # pylint: disable=arguments-differ
        """
        Set the arrival rate and routing matrix.
        Parameters:
            arrival_rate: arrival rate of customers.
            R: routing matrix, dim (m + 1 x m + 1), where m is number of nodes.

            For example:
            R[0, 0] is transition frome source to first node.
            R[0, m] is transition from source to out of system.

        """
        self.arrival_rate = arrival_rate
        self.R = R
        self.is_sources_set = True

    def set_nodes(self, b: list[list[float]], n: list[int]):  # pylint: disable=arguments-differ
        """
        Set the service time distribution parameters and number of channels for each node.
        Parameters:
            b: list of theoretical moments of service time distribution for each node.
            n: list of number of channels in each node.
        """
        self.b = b
        self.n = n
        self.is_nodes_set = True

    def balance_equation(self, L, R):
        """
        Calc the balance equation for the network.
        """
        # Create copies to avoid modifying original matrices
        L = np.array(L)
        R = np.array(R)

        # Identify null columns using vectorized operation
        null_mask = np.all(np.abs(R) < 1e-6, axis=0)
        _null_numbers = np.where(null_mask)[0]

        # Remove null columns and corresponding rows
        R_nonull = R[:, ~null_mask]
        R_nonull = R_nonull[~null_mask, :]

        n_rows_mod, n_cols_mod = R_nonull.shape

        # Calculate b and Q matrices using efficient indexing
        if n_rows_mod > 0:
            b = np.dot(L, R_nonull[0, :-1])
            Q = R_nonull[1:, :-1]
            A = np.eye(n_cols_mod - 1) - Q.T
            intensities = np.linalg.solve(A, b)
        else:
            intensities = np.array([], dtype=np.float64)

        # Create output array with zeros for null columns
        l_out = np.zeros(R.shape[1] - 1, dtype=np.float64)

        # Fill non-null columns
        _valid_indices = np.arange(len(intensities))
        l_out[~null_mask[:-1]] = intensities.flatten()

        return l_out

    def run(self) -> NetworkResults:
        """
        Run the simulation and calculate the results.
        """

        start = time.process_time()

        self._check_sources_and_nodes_is_set()

        nodes = len(self.n)
        loads = [0.0] * nodes
        v = []

        intensities = self.balance_equation(self.arrival_rate, self.R)
        intensities = [float(intensity) for intensity in intensities]

        v_node = []
        for i in range(nodes):
            node_arrival_rate = intensities[i]
            b1_node = self.b[i][0]

            loads[i] = node_arrival_rate * b1_node / self.n[i]

            if self.is_markovian:
                mmnr_calc = MMnrCalc(n=self.n[i], r=100)
                mmnr_calc.set_sources(l=node_arrival_rate)
                mmnr_calc.set_servers(mu=1 / b1_node)

                v_node.append(mmnr_calc.get_v())
            else:
                mgn_calc = MGnCalc(n=self.n[i])
                mgn_calc.set_sources(l=node_arrival_rate)
                mgn_calc.set_servers(b=self.b[i])

                mgn_calc.run()
                v_node.append(mgn_calc.get_v())

        h = 0.0001
        s = [h * (i + 1) for i in range(4)]

        I = np.eye(nodes)
        N = np.zeros((nodes, nodes))
        P = self.R[0, :nodes].reshape(1, -1)
        T = self.R[1:, nodes].reshape(-1, 1)
        Q = self.R[1:, :nodes]

        gamma_mu_alpha = [GammaDistribution.get_params([v_node[i][0], v_node[i][1]]) for i in range(nodes)]

        g_PLS = []
        for i in range(4):
            N = np.diag([lst_gamma(gamma_mu_alpha[j], s[i]) for j in range(nodes)])

            G = np.dot(N, Q)
            FF = I - G
            F = np.linalg.inv(FF)
            F = np.dot(P, np.dot(F, np.dot(N, T)))
            g_PLS.append(F[0, 0])

        v = diff5dots(g_PLS, h)
        v[0] = -v[0]
        v[2] = -v[2]

        v = [float(v) for v in v]
        loads = [float(l) for l in loads]

        self.results = NetworkResults(v=v, loads=loads, intensities=intensities, duration=time.process_time() - start)

        return self.results
