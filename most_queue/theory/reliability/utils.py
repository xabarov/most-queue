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
    for i, j, rate in transitions:
        if rate <= 0.0 or i == j:
            continue
        # build Q^T directly: entry (j, i) = q_ij
        rows.append(j)
        cols.append(i)
        vals.append(rate)
        diag[i] -= rate
    for i in range(n_states):
        rows.append(i)
        cols.append(i)
        vals.append(diag[i])

    # Replace the last equation with the normalization sum(pi) = 1
    a = csc_matrix((vals, (rows, cols)), shape=(n_states, n_states)).tolil()
    a[n_states - 1, :] = 1.0
    rhs = np.zeros(n_states)
    rhs[n_states - 1] = 1.0
    pi = spsolve(a.tocsc(), rhs)
    pi = np.maximum(pi, 0.0)
    return pi / pi.sum()
