"""
Calculates queueing network.
"""
import numpy as np

from most_queue.rand_distribution import GammaDistribution
from most_queue.theory.priority.mgn_invar_approx import MGnInvarApproximation
from most_queue.theory.utils.diff5dots import diff5dots
from most_queue.theory.utils.transforms import lst_gamma


class OpenNetworkCalcPriorities:
    """
    Calculates queueing network.
    """

    def __init__(self, R: list[np.matrix], b: list[list[list[float]]],
                 n: list[int], L: list[float], prty: list[str], nodes_prty: list[list[int]]):
        """
        R: list of routing matrices for each class.
        b: list of lists of theoretical moments of service time distribution for each class in each node.
        n: list of number of channels in each node.
        L: list of arrival intensities for each class.

        prty: list of priority types for each node. 
            No  - no priorities, FIFO
            PR  - preemptive resume, with resuming interrupted request
            RS  - preemptive repeat with resampling, re-sampling duration for new service
            RW  - preemptive repeat without resampling, repeating service with previous duration
            NP  - non preemptive, relative priority

        nodes_prty: Priority distribution among requests for each node in the network [m][x1, x2 .. x_k],
            m - node number, xi - priority for i-th class, k - number of classes
            For example: 
                [0][0,1,2] - for the first node, a direct order of priorities is set,
                [2][0,2,1] - for the third node, such an order of priorities is set: for the first class - the oldest (0),
                            for the second - the youngest (2), for the third - intermediate (1)
        """
        self.R = R
        self.b = b
        self.n = n
        self.L = L
        self.prty = prty
        self.nodes_prty = nodes_prty

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

        intensities = [self.balance_equation(
            self.L[k], self.R[k]) for k in range(k_num)]

        b_order = []
        l_order = []

        for m in range(nodes):
            orders = [self.nodes_prty[m][k] for k in range(k_num)]
            b_order.append([self.b[o][m] for o in orders])
            l_order.append([intensities[o][m] for o in orders])

        return b_order, l_order

    def run(self):
        """
        Run the simulation and calculate the results.
        """
        res = {}

        k_num = len(self.L)
        nodes = self.R[0].shape[0] - 1
        res['loads'] = [0.0] * nodes
        res['v'] = []

        b_order, l_order = self.order_b_l(k_num, nodes)

        res['v_node'] = []
        for i in range(nodes):
            l_sum = np.sum(l_order[i])
            b_sr = np.mean(b_order[i], axis=0)

            res['loads'][i] = l_sum * b_sr[0] / self.n[i]
            invar_calc = MGnInvarApproximation(
                l_order[i], b_order[i], self.n[i])
            res['v_node'].append(invar_calc.get_v(priority=self.prty[i]))
            for k in range(k_num):
                res['v_node'][i][self.nodes_prty[i][k]] = res['v_node'][i][k]

        h = 0.0001
        s = [h * (i + 1) for i in range(4)]

        for k in range(k_num):
            I = np.eye(nodes)
            N = np.zeros((nodes, nodes))
            P = self.R[k][0, :nodes].reshape(1, -1)
            T = self.R[k][1:, nodes].reshape(-1, 1)
            Q = self.R[k][1:, :nodes]

            gamma_mu_alpha = [GammaDistribution.get_params(
                [res['v_node'][i][k][0], res['v_node'][i][k][1]]) for i in range(nodes)]

            g_PLS = []
            for i in range(4):
                N = np.diag([lst_gamma(
                    gamma_mu_alpha[j], s[i]) for j in range(nodes)])

                G = np.dot(N, Q)
                FF = I - G
                F = np.linalg.inv(FF)
                F = np.dot(P, np.dot(F, np.dot(N, T)))
                g_PLS.append(F[0, 0])

            res['v'].append([])
            res['v'][k] = diff5dots(g_PLS, h)
            res['v'][k][0] = -res['v'][k][0]
            res['v'][k][2] = -res['v'][k][2]

        return res
