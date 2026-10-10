"""
Shared tagged-customer waiting-time construction for queueing-inventory models
(EPIC-075, item R1 of docs/roadmaps/literature_catchup_roadmap.md).

Literature note. That the waiting-time DISTRIBUTION (not only its mean) is
obtainable for queueing-inventory systems is established theory, not a result
of this library:

- Jeganathan K., Harikrishnan T., Selvakumar S., Anbazhagan N., Amutha S. et al.
  "Analysis of Interconnected Arrivals on Queueing-Inventory System with Two
  Multi-Server Service Channels and One Retrial Facility", Electronics 10(5):576,
  2021, doi:10.3390/electronics10050576 -- derives the LST of the unconditional
  waiting time via the matrix-geometric technique.
- Keerthana M., Sangeetha N., Sivakumar B., "Optimal service rates of a queueing
  inventory system with finite waiting hall, arbitrary service times and positive
  lead times", Annals of Operations Research 331(2):739-762, 2023.
- Surveys: Krishnamoorthy A., Lakshmy B., Manikandan R., OPSEARCH 48:153-169,
  2011, doi:10.1007/s12597-010-0032-z; Krishnamoorthy A., Shajin D.,
  Narayanan V.C., "Inventory with Positive Service Time: a Survey", in Queueing
  Theory 2, Wiley, 2021, pp. 201-237, doi:10.1002/9781119755234.ch6.

What this module contributes is the IMPLEMENTATION, by a different computational
route: a tagged-customer absorbing chain solved by matrix-exponential action (the
technique used throughout this library's batch-service family) instead of
numerical Laplace inversion. That route gives numerically stable values deep in
the tail, where inverting an LST is delicate.

The construction. A tagged arrival that finds ``n`` customers in system and
stock ``i`` begins service only once BOTH conditions hold: fewer than ``c``
customers ahead are still present (a server is free) AND stock is positive.
The second condition is what distinguishes this from an ordinary M/M/c wait --
a stockout keeps a customer waiting even with an empty queue. Transient state
is ``(r, i)``: ``r`` customers still ahead, ``i`` units of stock. Arrivals
*behind* the tagged customer never affect its wait under FCFS and are not
tracked, which is what keeps the chain small.

Every state with ``r < c`` and ``i == 0`` behaves identically -- no service can
complete while stock is out, so the only event is a replenishment, which
immediately frees the tagged customer. They are therefore collapsed into a
single "stocked out, server free" state.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_wait_phase_type(
    *,
    level_phase_probs,
    c: int,
    s_max: int,
    s: int,
    mu: float,
    theta: float,
    n_max: int,
    lost_sales: bool,
):
    """
    Build the phase-type representation of the waiting time.

    :param level_phase_probs: callable ``n -> array of length s_max+1`` giving
        the stationary probabilities ``pi(n, i)`` of ``n`` customers in system
        and stock level ``i``.
    :param c: number of (identical) servers; the tagged customer needs fewer
        than ``c`` customers ahead before a server is free.
    :param s_max: S, the stock level restored on replenishment.
    :param s: reorder point; an order is in transit whenever stock <= s.
    :param mu: per-server service rate.
    :param theta: replenishment rate (Exp lead time).
    :param n_max: truncation on the number of customers found on arrival.
    :param lost_sales: if True, an arrival finding zero stock is turned away and
        never waits, so it must not enter the initial vector.
    :return: ``(alpha, T, p_zero)`` -- initial vector over transient states, the
        sparse sub-generator, and the probability of zero wait.
    """
    m = s_max + 1

    # Transient states: (r, i) for r = c..n_max and any i, plus one collapsed
    # "stocked out with a free server" state.
    def idx(r, i):
        return (r - c) * m + i

    n_transient_levels = max(n_max - c + 1, 0)
    stuck = n_transient_levels * m
    size = stuck + 1

    rows, cols, vals = [], [], []
    out = np.zeros(size)

    def add(src, dst, rate):
        rows.append(src)
        cols.append(dst)
        vals.append(rate)
        out[src] += rate

    for r in range(c, n_max + 1):
        for i in range(m):
            src = idx(r, i)
            if i >= 1:
                rate = min(r, c) * mu  # busy servers among those ahead
                if r - 1 >= c:
                    add(src, idx(r - 1, i - 1), rate)
                elif i - 1 >= 1:
                    out[src] += rate  # absorbed: a server frees up and stock remains
                else:
                    add(src, stuck, rate)  # server free but stock just ran out
            if i <= s:
                add(src, idx(r, s_max), theta)
    # From the collapsed state only a replenishment can occur, and it absorbs
    # at once (stock becomes S >= 1 while a server is already free).
    out[stuck] += theta

    q = sp.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()
    t_mat = (q - sp.diags(out)).tocsc()

    alpha = np.zeros(size)
    p_zero = 0.0
    accepted = 0.0
    for n in range(n_max + 1):
        probs = level_phase_probs(n)
        for i in range(m):
            prob = float(probs[i])
            if prob <= 0:
                continue
            if lost_sales and i == 0:
                continue  # turned away, never waits
            accepted += prob
            if n < c:
                if i >= 1:
                    p_zero += prob  # free server and stock: served at once
                else:
                    alpha[stuck] += prob
            else:
                alpha[idx(n, i)] += prob
    if accepted <= 0:
        raise ValueError("no accepted-arrival probability mass; check parameters")
    alpha /= accepted
    p_zero /= accepted
    return alpha, t_mat, p_zero


def phase_type_moments(alpha: np.ndarray, t_mat, num: int) -> list[float]:
    """Raw moments E[W^k] = k! * alpha (-T)^{-k} 1, k = 1..num."""
    vec = np.ones(t_mat.shape[0])
    moments: list[float] = []
    factorial = 1.0
    for k in range(1, num + 1):
        vec = spla.spsolve((-t_mat).tocsc(), vec)
        factorial *= k
        moments.append(float(factorial * (alpha @ vec)))
    return moments


def phase_type_tail(alpha: np.ndarray, t_mat, t: float) -> float:
    """P(W > t) = alpha exp(T t) 1, computed by matrix-exponential action."""
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")
    if t == 0:
        return float(alpha.sum())
    vec = spla.expm_multiply(t_mat.T.tocsc() * t, alpha)
    return float(np.sum(vec))
