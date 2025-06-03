"""
Calculates queueing network.
"""
import numpy as np

from most_queue.rand_distribution import GammaDistribution
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.theory.utils.diff5dots import diff5dots
from most_queue.theory.utils.transforms import lst_gamma


class OpenNetworkCalc:
    """
    Calculates queueing network.
    """

    def __init__(self, R: np.matrix, b: list[list[float]],
                 n: list[int], arrival_rate: float):
        """
        R: routing matrix
        b: list of theoretical moments of service time distribution for each node.
        n: list of number of channels in each node.
        L: arrival intensitiy.
        """
        self.R = R
        self.b = b
        self.n = n
        self.arrival_rate = arrival_rate

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

    def run(self, is_markovian=False):
        """
        Run the simulation and calculate the results.
        """
        res = {}

        nodes = len(self.n)
        res['loads'] = [0.0] * nodes
        res['v'] = []

        intensities = self.balance_equation(self.arrival_rate, self.R)

        res['v_node'] = []
        for i in range(nodes):
            node_arrival_rate = intensities[i]
            b1_node = self.b[i][0]

            res['loads'][i] = node_arrival_rate * b1_node / self.n[i]
            
            if is_markovian:
                mmnr_calc = MMnrCalc(l=node_arrival_rate, mu=1/b1_node, n=self.n[i], r=100)
                res['v_node'].append(mmnr_calc.get_v())
            else:
                mgn_calc = MGnCalc(n=self.n[i], l=node_arrival_rate, b=self.b[i])
                mgn_calc.run()
                res['v_node'].append(mgn_calc.get_v())

        h = 0.0001
        s = [h * (i + 1) for i in range(4)]

        I = np.eye(nodes)
        N = np.zeros((nodes, nodes))
        P = self.R[0, :nodes].reshape(1, -1)
        T = self.R[1:, nodes].reshape(-1, 1)
        Q = self.R[1:, :nodes]

        gamma_mu_alpha = [GammaDistribution.get_params(
            [res['v_node'][i][0], res['v_node'][i][1]]) for i in range(nodes)]

        g_PLS = []
        for i in range(4):
            N = np.diag([lst_gamma(
                gamma_mu_alpha[j], s[i]) for j in range(nodes)])

            G = np.dot(N, Q)
            FF = I - G
            F = np.linalg.inv(FF)
            F = np.dot(P, np.dot(F, np.dot(N, T)))
            g_PLS.append(F[0, 0])

        res['v'] = diff5dots(g_PLS, h)
        res['v'][0] = -res['v'][0]
        res['v'][2] = -res['v'][2]
        
        res['intensities'] = intensities
        res['v'] = [float(v) for v in res['v']]
        res['loads'] = [float(l) for l in res['loads']]


        return res
