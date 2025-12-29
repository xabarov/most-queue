"""
Calculates queueing network with negative jobs.
"""

import time

import numpy as np

from most_queue.random.distributions import GammaDistribution
from most_queue.sim.negative import NegativeServiceType
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc
from most_queue.theory.networks.base_network_calc import BaseNetwork, NetworkResults
from most_queue.theory.utils.diff5dots import diff5dots
from most_queue.theory.utils.transforms import lst_gamma


class NegativeNetworkCalc(BaseNetwork):
    """
    Calculates queueing network with negative jobs.
    """

    def __init__(self, negative_arrival_type: str = "global"):
        """
        Initializes the NegativeNetworkCalc class
        :param negative_arrival_type: "global" or "per_node" - type of negative arrival distribution
        """
        super().__init__()

        self.R = None
        self.b = None
        self.n = None
        self.arrival_rate = None
        self.negative_arrival_type = negative_arrival_type
        self.negative_arrival_rate = None  # For global type
        self.negative_arrival_rates = None  # For per_node type
        self.negative_types = None

    def set_sources(
        self,
        arrival_rate: float,
        R: np.matrix,
        negative_arrival_rate: float = None,
        negative_arrival_rates: list[float] = None,
    ):  # pylint: disable=arguments-differ
        """
        Set the arrival rate and routing matrix.
        Parameters:
            arrival_rate: arrival rate of positive customers.
            R: routing matrix, dim (m + 1 x m + 1), where m is number of nodes.
                For example:
                R[0, 0] is transition from source to first node.
                R[0, m] is transition from source to out of system.
            negative_arrival_rate: arrival rate of negative customers for global type (optional).
                Used only if negative_arrival_type is "global".
            negative_arrival_rates: list of arrival rates for negative customers per node (optional).
                List should have length equal to number of nodes.
                Used only if negative_arrival_type is "per_node".
        """
        self.arrival_rate = arrival_rate
        self.R = R

        if self.negative_arrival_type == "global":
            self.negative_arrival_rate = negative_arrival_rate if negative_arrival_rate is not None else 0.0
            self.negative_arrival_rates = None
        elif self.negative_arrival_type == "per_node":
            if negative_arrival_rates is not None:
                # Validate length if nodes are already set
                if hasattr(self, "n") and self.n is not None:
                    if len(negative_arrival_rates) != len(self.n):
                        raise ValueError(
                            f"negative_arrival_rates length ({len(negative_arrival_rates)}) "
                            f"must match number of nodes ({len(self.n)})"
                        )
                self.negative_arrival_rates = negative_arrival_rates
            else:
                # Default to zero for all nodes
                if hasattr(self, "n") and self.n is not None:
                    self.negative_arrival_rates = [0.0] * len(self.n)
                else:
                    self.negative_arrival_rates = None
            self.negative_arrival_rate = None
        else:
            raise ValueError(f"Unknown negative_arrival_type: {self.negative_arrival_type}")

        self.is_sources_set = True

    def set_nodes(
        self,
        b: list[list[float]],
        n: list[int],
        negative_types: list[NegativeServiceType],
    ):  # pylint: disable=arguments-differ
        """
        Set the service time distribution parameters and number of channels for each node.
        Parameters:
            b: list of theoretical moments of service time distribution for each node.
            n: list of number of channels in each node.
            negative_types: list of NegativeServiceType for each node (DISASTER or RCS).
        """
        self.b = b
        self.n = n
        self.negative_types = negative_types

        # Validate negative types
        for i, neg_type in enumerate(negative_types):
            if neg_type not in [NegativeServiceType.DISASTER, NegativeServiceType.RCS]:
                raise ValueError(f"Node {i}: Only DISASTER and RCS negative types are supported, got {neg_type}")

        # Validate per_node negative_arrival_rates length if needed
        if (
            self.negative_arrival_type == "per_node"
            and self.negative_arrival_rates is not None
            and len(self.negative_arrival_rates) != len(n)
        ):
            raise ValueError(
                f"negative_arrival_rates length ({len(self.negative_arrival_rates)}) "
                f"must match number of nodes ({len(n)})"
            )

        self.is_nodes_set = True

    def balance_equation(self, L, R):
        """
        Calc the balance equation for the network.
        Same as in OpenNetworkCalc - calculates positive arrival intensities.
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
        Run the calculation and calculate the results.
        """
        start = time.process_time()

        self._check_sources_and_nodes_is_set()

        nodes = len(self.n)
        loads = [0.0] * nodes
        v = []

        # Calculate positive arrival intensities using balance equations
        intensities = self.balance_equation(self.arrival_rate, self.R)
        intensities = [float(intensity) for intensity in intensities]

        # Calculate negative arrival rates for each node
        if self.negative_arrival_type == "global":
            negative_rates = [self.negative_arrival_rate] * nodes
        else:  # per_node
            if self.negative_arrival_rates is None:
                negative_rates = [0.0] * nodes
            else:
                negative_rates = self.negative_arrival_rates

        v_node = []
        for i in range(nodes):
            node_arrival_rate = intensities[i]
            b1_node = self.b[i][0]

            loads[i] = node_arrival_rate * b1_node / self.n[i]

            # Determine which calculator to use based on negative type
            neg_type = self.negative_types[i]
            l_neg = negative_rates[i]

            if neg_type == NegativeServiceType.DISASTER:
                calc = MGnNegativeDisasterCalc(n=self.n[i])
            elif neg_type == NegativeServiceType.RCS:
                calc = MGnNegativeRCSCalc(n=self.n[i])
            else:
                raise ValueError(f"Unsupported negative type: {neg_type}")

            # Set sources and servers
            calc.set_sources(l_pos=node_arrival_rate, l_neg=l_neg)
            calc.set_servers(b=self.b[i])

            # Run calculation
            calc.run()

            # Get sojourn time moments
            node_results = calc.get_results()
            v_node.append(node_results.v)

        # Recompute network-level results using flow decomposition
        # Similar to OpenNetworkCalc.run()
        h = 0.0001
        s = [h * (i + 1) for i in range(4)]

        I = np.eye(nodes)
        N = np.zeros((nodes, nodes))
        P = self.R[0, :nodes].reshape(1, -1)
        T = self.R[1:, nodes].reshape(-1, 1)
        Q = self.R[1:, :nodes]

        # Approximate node sojourn time distributions with gamma
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
