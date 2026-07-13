"""
Traffic (flow balance) equations for open queueing networks.
"""

import numpy as np


def solve_traffic_equations(arrival_rate: float, R) -> np.ndarray:
    """
    Solve the flow balance equations of an open network.

    :param arrival_rate: external (source) arrival rate.
    :param R: routing matrix, dim (m + 1 x m + 1), where m is the number of
        nodes: row 0 — transitions from the source to nodes, last column —
        transitions out of the system.
    :return: arrival intensities into each node, shape (m,).
    """
    R = np.asarray(R, dtype=float)
    m = R.shape[0] - 1
    b = arrival_rate * R[0, :m]
    Q = R[1:, :m]  # node-to-node transition probabilities
    A = np.eye(m) - Q.T
    return np.linalg.solve(A, b)
