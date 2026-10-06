"""
Shared "idle-refill race" subgenerator for general bulk-service rule (1 <= a <= b).

EPIC-067: when a tagged customer's own batch cannot start the instant the
batches ahead of it clear (the post-service remainder is below the threshold
``a``), the wait is governed by a RACE between new Poisson(lambda) arrivals
and the sequence of remaining service phases -- NOT a simple sequential
"service phase, then refill phase" split (that naive version was confirmed
numerically wrong: 17-30% error on specific states against an independent
simulator, see docs/epics/EPIC-067-bulk-service-idle-refill.md).

``need`` extra arrivals must land before the given sequence of service phase
rates finishes, or a pure Erlang(need - count, lambda) refill tail follows.
Reused by ``BulkServiceMM1Calc`` (a 1-rate-then-another-rate sequence) and
``BulkServiceErlangCalc`` (a k-p-then-k*full_ahead-phase sequence); H2's
branching chain needs its own construction (the race interacts with which
branch is active), see ``bulk_service_h2.py``.
"""

import numpy as np
import scipy.sparse as sp


def race_subgen(service_rates: np.ndarray, lam: float, need: int):
    """
    Build the race-aware absorbing-chain subgenerator.

    :param service_rates: rates of the sequential service phases still ahead of
        the tagged customer's own batch starting (e.g. remaining current-batch
        phases followed by full-batches-ahead phases), in order.
    :param lam: arrival rate.
    :param need: extra Poisson(lam) arrivals needed (``a - remainder - 1``) for
        the tagged customer's own batch to reach threshold ``a`` once the
        service phases above finish. Must be > 0 (callers use the plain
        sequential chain directly when ``need <= 0``).
    :return: (subgen, m, start) -- sparse subgenerator, its size, and the
        index of the starting (first service phase, zero arrivals) state.
    """
    n_sp = len(service_rates)
    n_grid = n_sp * (need + 1)  # (service phase, arrivals-so-far) grid
    m = n_grid + need  # + refill tail states (need, need-1, ..., 1)
    rows, cols, vals = [], [], []
    out_rate = np.zeros(m)

    def grid(phase, count):
        return phase * (need + 1) + count

    def refill(remaining):  # remaining in 1..need
        return n_grid + (remaining - 1)

    for phase in range(n_sp):
        r_s = service_rates[phase]
        for count in range(need + 1):
            s = grid(phase, count)
            if count < need:
                rows.append(s)
                cols.append(grid(phase, count + 1))
                vals.append(lam)
                out_rate[s] += lam
            if phase < n_sp - 1:
                rows.append(s)
                cols.append(grid(phase + 1, count))
                vals.append(r_s)
                out_rate[s] += r_s
            elif count == need:
                out_rate[s] += r_s  # enough already accumulated -> absorbs directly
            else:
                rows.append(s)
                cols.append(refill(need - count))
                vals.append(r_s)
                out_rate[s] += r_s
    for remaining in range(1, need + 1):
        s = refill(remaining)
        if remaining > 1:
            rows.append(s)
            cols.append(refill(remaining - 1))
            vals.append(lam)
        out_rate[s] += lam  # remaining==1 absorbs directly

    q = sp.coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
    subgen = (q - sp.diags(out_rate)).tocsc()
    return subgen, m, grid(0, 0)
