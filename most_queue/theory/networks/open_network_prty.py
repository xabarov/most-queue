"""
Calculates queueing network.
"""

import time

import numpy as np

from most_queue.distributions import GammaDistribution
from most_queue.theory.networks.base_network_calc import (
    BaseNetworkPriority,
    NetworkResultsPriority,
)
from most_queue.theory.priority.mgn_invar_approx import MGnInvarApproximation
from most_queue.theory.utils.diff5dots import diff5dots
from most_queue.theory.utils.transforms import lst_gamma


class OpenNetworkCalcPriorities(BaseNetworkPriority):
    """
    Calculates queueing network.
    """

    def __init__(self):
        """
        Initializes the network calculator.
        """

        super().__init__()
        self.R = None
        self.b = None
        self.n = None
        self.L = None
        self.prty = None
        self.nodes_prty = None
        self.is_sources_set = False
        self.is_nodes_set = False
        self.intensities = None

    def set_sources(self, L: list[float], R: list[np.matrix]):  # pylint: disable=arguments-differ
        """
        R: list of routing matrices for each class.
        L: list of arrival intensities for each class.
        each routing matrix, dim (m + 1 x m + 1), where m is number of nodes.
            For example:
            R[0][0, 0] is transition frome source to first node for the first class.
            R[0][0, m] is transition from source to out of system for the first class.

        """
        self.L = L
        self.R = R
        self.is_sources_set = True

    def set_nodes(
        self,
        b: list[list[list[float]]],
        n: list[int],
        prty: list[str],
        nodes_prty: list[list[int]],
    ):  # pylint: disable=arguments-differ
        """
        Set nodes of queueing network.
        Parameters:
        b: list of lists of theoretical moments of service time distribution
        for each class in each node.
        n: list of number of channels in each node.
        prty: list of priority types for each node.
            No  - no priorities, FIFO
            PR  - preemptive resume, with resuming interrupted request
            RS  - preemptive repeat with resampling, re-sampling duration for new service
            RW  - preemptive repeat without resampling, repeating service with previous duration
            NP  - non preemptive, relative priority

        nodes_prty: Priority distribution among requests for each
        node in the network [m][x1, x2 .. x_k],
            m - node number, xi - priority for i-th class, k - number of classes
            For example:
                [0][0,1,2] - for the first node, a direct order of priorities is set,
                [2][0,2,1] - for the third node, such an order of
                priorities is set: for the first class - the oldest (0),
                for the second - the youngest (2), for the third - intermediate (1)
        """
        self.b = b
        self.n = n
        self.prty = prty
        self.nodes_prty = nodes_prty
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

    def order_b_l(self, k_num, nodes):
        """
        Order the loads and intensities based on node priority.
        """

        self.intensities = [self.balance_equation(self.L[k], self.R[k]) for k in range(k_num)]

        b_order = []
        l_order = []

        for m in range(nodes):
            orders = [self.nodes_prty[m][k] for k in range(k_num)]
            b_order.append([self.b[o][m] for o in orders])
            l_order.append([self.intensities[o][m] for o in orders])

        return b_order, l_order

    def run(self) -> NetworkResultsPriority:
        """
        Run the simulation and calculate the results.
        """

        start = time.process_time()

        self._check_sources_and_nodes_is_set()

        k_num = len(self.L)
        nodes = self.R[0].shape[0] - 1
        loads = [0.0] * nodes
        v = []

        b_order, l_order = self.order_b_l(k_num, nodes)

        v_node = []
        for i in range(nodes):
            l_sum = np.sum(l_order[i])
            b_sr = np.mean(b_order[i], axis=0)

            loads[i] = l_sum * b_sr[0] / self.n[i]
            invar_calc = MGnInvarApproximation(n=self.n[i], priority=self.prty[i])
            invar_calc.set_sources(l_order[i])
            invar_calc.set_servers(b_order[i])
            v_node.append(invar_calc.get_v())
            for k in range(k_num):
                v_node[i][self.nodes_prty[i][k]] = v_node[i][k]

        h = 0.0001
        s = [h * (i + 1) for i in range(4)]

        for k in range(k_num):
            I = np.eye(nodes)
            N = np.zeros((nodes, nodes))
            P = self.R[k][0, :nodes].reshape(1, -1)
            T = self.R[k][1:, nodes].reshape(-1, 1)
            Q = self.R[k][1:, :nodes]

            gamma_mu_alpha = [GammaDistribution.get_params([v_node[i][k][0], v_node[i][k][1]]) for i in range(nodes)]

            g_PLS = []
            for i in range(4):
                N = np.diag([lst_gamma(gamma_mu_alpha[j], s[i]) for j in range(nodes)])

                G = np.dot(N, Q)
                FF = I - G
                F = np.linalg.inv(FF)
                F = np.dot(P, np.dot(F, np.dot(N, T)))
                g_PLS.append(F[0, 0])

            v.append([])
            v[k] = diff5dots(g_PLS, h)
            v[k][0] = -v[k][0]
            v[k][2] = -v[k][2]

        self.results = NetworkResultsPriority(
            v=v, intensities=self.intensities, loads=loads, duration=time.process_time() - start
        )

        return self.results
