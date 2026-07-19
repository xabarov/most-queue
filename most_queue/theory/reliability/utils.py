"""
Shared helper for the reliability models: stationary distribution of a finite
(truncated) CTMC given its transition rates, via a sparse linear solve.
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def ctmc_stationary(transitions, n_states: int) -> np.ndarray:
    """
    Solve pi Q = 0, sum(pi) = 1 for a finite CTMC.

    :param transitions: iterable of (from_state, to_state, rate), rate > 0.
    :param n_states: number of states.
    :return: stationary probabilities, shape (n_states,).
    """
    rows, cols, vals = [], [], []
    diag = np.zeros(n_states)
    last = n_states - 1
    for i, j, rate in transitions:
        if rate <= 0.0 or i == j:
            continue
        # build Q^T directly: entry (j, i) = q_ij; the last row of the
        # system is replaced by the normalization, so skip entries there
        if j != last:
            rows.append(j)
            cols.append(i)
            vals.append(rate)
        diag[i] -= rate
    for i in range(last):
        rows.append(i)
        cols.append(i)
        vals.append(diag[i])

    # Normalization row: sum(pi) = 1
    rows.extend([last] * n_states)
    cols.extend(range(n_states))
    vals.extend([1.0] * n_states)

    a = csc_matrix((vals, (rows, cols)), shape=(n_states, n_states))
    rhs = np.zeros(n_states)
    rhs[last] = 1.0
    pi = spsolve(a, rhs)
    pi = np.maximum(pi, 0.0)
    return pi / pi.sum()
