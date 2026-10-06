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


def abandonment_chain(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    rate_first: float,
    n_first: int,
    rate_ahead: float,
    k: int,
    j: int,
    a: int,
    b: int,
    lam: float,
    gamma: float,
    start_idle: bool = False,
):
    """
    EPIC-068: race-aware absorbing chain for a tagged customer's wait when
    EVERY waiting customer (including those ahead of the tagged one) may
    independently abandon at rate ``gamma`` (Markovian/memoryless patience,
    the same convention as ``most_queue.theory.impatience.mm1.MM1Impatience``).

    A naive "ignore the ahead customers' abandonment, just add a standalone
    competing gamma-exit to the EPIC-067 chain" hypothesis was confirmed
    numerically wrong (independent per-state simulation): the ahead-of-tagged
    count is NOT the deterministic ``j`` of EPIC-067 anymore -- it is itself a
    pure-death process (each of the ``R`` originally-ahead survivors abandons
    independently) running CONCURRENTLY with batch formation, so it must be
    tracked as an explicit state dimension, not folded into a fixed
    "full_ahead/remainder" split.

    Key structural fact (proved, not just assumed) that keeps this tractable:
    whenever a batch forms ahead of the tagged customer WITHOUT including it,
    that batch's size is always EXACTLY ``b`` (if it were less than ``b`` and
    still excluded the tagged customer, the tagged customer plus the survivors
    would already be <= the batch cap, contradiction). So -- exactly as in
    EPIC-067 -- every such "ahead" batch uses ``rate_ahead`` (``rate_fn(b)``);
    only the VERY FIRST segment (the batch the tagged customer actually
    observed on arrival, part-way through) uses ``rate_first``
    (``rate_fn(i_observed)``) for its remaining ``n_first = k - p_observed``
    phases. No other batch-size-dependent rate bookkeeping is needed.

    State: (segment in {first, ahead}, phase within that segment's k-cycle,
    R = surviving originally-ahead count 0..j, K = new arrivals behind the
    tagged customer accumulated so far, capped at ``a-1``), plus idle/refill
    states (R, K) with no phase. R decreases via abandonment (rate R*gamma)
    and via being swept into a full ``b``-sized batch; K increases via new
    arrivals (rate lam) and resets implicitly whenever consumed into a batch
    that reaches the tagged customer (irrelevant once absorbed).

    :return: ``(subgen, m, start, serve_rate)`` -- the sparse subgenerator
        (its diagonal already includes the universal ``-gamma`` competing
        exit), its size, the index of the starting state, and a length-``m``
        array giving the "tagged customer gets served" absorption RATE out of
        each state (0 for non-absorbing states). ``gamma * alpha @ (-A)^-1 @
        1`` gives ``P(abandon)``; ``n! * alpha @ (-A)^-(n+1) @ serve_rate``
        gives the raw moments of ``W`` restricted to the "served" outcome
        (divide by ``P(served) = 1 - P(abandon)`` for the conditional
        moments).
    """
    n_states_per_phase = (j + 1) * a

    def idx_first(phase, r_val, k_val):
        return phase * n_states_per_phase + r_val * a + k_val

    base_ahead = n_first * n_states_per_phase

    def idx_ahead(phase, r_val, k_val):
        return base_ahead + phase * n_states_per_phase + r_val * a + k_val

    base_idle = base_ahead + k * n_states_per_phase

    def idx_idle(r_val, k_val):
        return base_idle + r_val * a + k_val

    m = base_idle + n_states_per_phase
    rows, cols, vals = [], [], []
    out_rate = np.zeros(m)
    serve_rate = np.zeros(m)

    def add(src, dst, rate):
        rows.append(src)
        cols.append(dst)
        vals.append(rate)
        out_rate[src] += rate

    def build_segment(idx_fn, n_phases, rate, next_idx_fn):
        for phase in range(n_phases):  # pylint: disable=too-many-nested-blocks
            for r_val in range(j + 1):
                for k_val in range(a):
                    s = idx_fn(phase, r_val, k_val)
                    if phase < n_phases - 1:
                        add(s, idx_fn(phase + 1, r_val, k_val), rate)
                    else:
                        total = r_val + 1 + k_val
                        if total >= a:
                            take = min(b, total)
                            if take >= r_val + 1:
                                out_rate[s] += rate
                                serve_rate[s] += rate
                            else:
                                add(s, next_idx_fn(0, r_val - take, k_val), rate)  # take == b, proved above
                        else:
                            add(s, idx_idle(r_val, k_val), rate)
                    if r_val >= 1:
                        add(s, idx_fn(phase, r_val - 1, k_val), r_val * gamma)
                    if k_val < a - 1:
                        add(s, idx_fn(phase, r_val, k_val + 1), lam)

    build_segment(idx_first, n_first, rate_first, idx_ahead)
    build_segment(idx_ahead, k, rate_ahead, idx_ahead)

    for r_val in range(j + 1):
        for k_val in range(a):
            s = idx_idle(r_val, k_val)
            total = r_val + 1 + k_val
            if total < a:
                if r_val >= 1:
                    add(s, idx_idle(r_val - 1, k_val), r_val * gamma)
                if k_val < a - 1:
                    if r_val + 1 + (k_val + 1) >= a:
                        out_rate[s] += lam
                        serve_rate[s] += lam
                    else:
                        add(s, idx_idle(r_val, k_val + 1), lam)

    for s in range(m):
        out_rate[s] += gamma  # tagged customer's own patience: universal competing exit

    # Idle/refill states are only ever used (as source or destination) when the
    # idle-refill regime is actually reachable (total = r_val+1+k_val < a somewhere);
    # e.g. at a=1 it never is, since any single arrival already meets the threshold.
    # Such states get zero out_rate at gamma==0 (no competing exit to fall back on,
    # unlike gamma>0 where the universal patience exit above keeps every row
    # non-degenerate) -- a fully isolated all-zero row, which makes (-A) exactly
    # singular for the moment formula's repeated spsolve (expm_multiply tolerates it,
    # which is why this stayed hidden: EPIC-068 only ever called this with gamma>0,
    # the first gamma==0 caller -- EPIC-069's constant-mu multiserver reuse -- hit it).
    # These rows are provably unreachable (nothing transitions into or out of them),
    # so pinning their out_rate to an arbitrary positive value changes nothing for any
    # reachable state's moments/tail/absorption probability.
    out_rate[out_rate == 0] = 1.0

    q = sp.coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
    subgen = (q - sp.diags(out_rate)).tocsc()
    start = idx_idle(j, 0) if start_idle else idx_first(0, j, 0)
    return subgen, m, start, serve_rate
