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

Heterogeneous servers need no extra machinery here, which is not obvious but is
easy to see: while the tagged customer waits there are always at least ``c``
customers ahead of it, so *every* server is busy and the aggregate completion
rate is simply ``sum(mu_j)`` -- which server is doing what never enters the
calculation. The busy-SUBSET dimension that the stationary solution needs for
``n < c`` is therefore irrelevant to the wait, and the only thing that changes
between the identical- and heterogeneous-server models is the value of
``full_rate`` (``c*mu`` versus ``sum(mu_j)``).

Two builders live here. :func:`build_wait_phase_type` is the scalar one above,
used by the exponential single- and multi-server models.
:func:`build_wait_phase_type_blocks` reads the dynamics off the QBD blocks of
the repeating part instead of taking a single rate, which is what the Erlang and
H2 service variants need (there the completion rate depends on which server sits
at which service phase). :class:`WaitPhaseTypeMixin` wires the second one into a
calculator.
"""

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_wait_phase_type(  # pylint: disable=too-many-arguments
    *,
    level_phase_probs,
    c: int,
    s_max: int,
    s: int,
    full_rate: float,
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
    :param full_rate: aggregate completion rate while ALL ``c`` servers are
        busy -- ``c * mu`` for identical servers, ``sum(mu_j)`` for
        heterogeneous ones. This is the only rate the wait ever sees (see the
        module docstring).
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
                # r >= c throughout the transient region, so every server is
                # busy and the aggregate rate is constant.
                rate = full_rate
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


def build_wait_phase_type_blocks(  # pylint: disable=too-many-arguments
    *,
    boundary_stock_probs,
    repeating_phase_probs,
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    m: int,
    c: int,
    theta: float,
    n_max: int,
    lost_sales: bool,
):
    """
    Block-based variant of :func:`build_wait_phase_type`, for models whose
    repeating part carries a phase beyond the stock level.

    Where the scalar builder above needs a single aggregate completion rate,
    this one reads the dynamics straight off the QBD blocks of the repeating
    part, so it covers heterogeneous servers with Erlang or H2 service times
    (where the completion rate depends on which server sits at which service
    phase) with no model-specific code.

    Two observations make it work:

    1. While the tagged customer waits there are always at least ``c``
       customers ahead of it, so every server is busy -- the system is in the
       REPEATING part of the QBD for the whole wait, and ``A0/A1/A2`` describe
       it completely.
    2. Arrivals behind the tagged customer never affect its wait under FCFS.
       An ``A0`` transition therefore keeps the tagged customer's level (the
       number of customers still AHEAD of it) unchanged while applying ``A0``'s
       phase change, which is why the within-level generator is ``A0 + A1``.
       Row sums of ``A0 + A1`` equal minus the row sums of ``A2``, so nothing
       has to be bookkept by hand.

    Absorption happens on the ``A2`` transition out of level ``c``: a server
    frees up, and the tagged customer enters service provided the completion
    left stock positive. If it took the last unit, the customer lands in the
    single collapsed "server free, stock out" state and waits out one
    replenishment.

    :param boundary_stock_probs: callable ``n -> array of length m`` with the
        stationary stock distribution at boundary level ``n < c`` (summed over
        busy-server configurations -- which servers are busy is irrelevant to a
        customer that will not wait at all).
    :param repeating_phase_probs: callable ``n -> array of length m_rep`` with
        the stationary distribution over repeating phases at level ``n >= c``.
    :param a0: repeating-part up-block (arrivals).
    :param a1: repeating-part local block.
    :param a2: repeating-part down-block (service completions).
    :param m: stock dimension ``s_max + 1``; repeating phase ``p`` carries
        stock ``p % m`` (the layout every model in this package uses --
        configuration index major, stock level minor).
    :param c: number of servers.
    :param theta: replenishment rate (Exp lead time).
    :param n_max: truncation on the number of customers found on arrival.
    :param lost_sales: if True, an arrival finding zero stock never waits.
    :return: ``(alpha, T, p_zero)``, as in :func:`build_wait_phase_type`.
    """
    m_rep = a1.shape[0]
    n_levels = max(n_max - c + 1, 0)
    if n_levels == 0:
        raise ValueError(f"level truncation {n_max} must be at least c={c}")

    def idx(r, p):
        return (r - c) * m_rep + p

    stuck = n_levels * m_rep
    size = stuck + 1

    within = a0 + a1
    rows, cols, vals = [], [], []

    def add(src, dst, rate):
        if rate != 0.0:
            rows.append(src)
            cols.append(dst)
            vals.append(rate)

    nz_within = np.argwhere(within != 0.0)
    nz_down = np.argwhere(a2 != 0.0)
    for r in range(c, n_max + 1):
        for p, q in nz_within:
            add(idx(r, p), idx(r, q), within[p, q])
        for p, q in nz_down:
            if r > c:
                add(idx(r, p), idx(r - 1, q), a2[p, q])
            elif q % m == 0:
                # The completing service took the last stock unit: a server is
                # free but cannot start the tagged customer until a delivery.
                add(idx(r, p), stuck, a2[p, q])
            # else: a server freed up with stock left -- absorbed, and the
            # outflow is already on within's diagonal, so nothing to add.
    # Only a replenishment can happen in the collapsed state, and it absorbs.
    add(stuck, stuck, -theta)

    t_mat = sp.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsc()

    alpha = np.zeros(size)
    p_zero = 0.0
    accepted = 0.0
    for n in range(n_max + 1):
        if n < c:
            probs = boundary_stock_probs(n)
            for i in range(m):
                prob = float(probs[i])
                if prob <= 0 or (lost_sales and i == 0):
                    continue
                accepted += prob
                if i >= 1:
                    p_zero += prob  # free server and stock: served at once
                else:
                    alpha[stuck] += prob
        else:
            probs = repeating_phase_probs(n)
            for p in range(m_rep):
                prob = float(probs[p])
                if prob <= 0 or (lost_sales and p % m == 0):
                    continue
                accepted += prob
                alpha[idx(n, p)] += prob
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


class WaitPhaseTypeMixin:
    """
    Exact waiting-time moments and tail for the heterogeneous-server
    queueing-inventory calculators, on top of :func:`build_wait_phase_type_blocks`.

    A host class supplies ``_boundary_stock_probs(n)`` -- the stationary stock
    distribution at boundary level ``n < c``, summed over busy-server
    configurations -- and must expose ``c``, ``s_max``, ``theta``, ``policy``
    and a solved QBD via ``_build_solver()``. Everything else (which servers
    are busy, which service phase each one is at) is read off the QBD blocks,
    so the same mixin serves exponential, Erlang and H2 service times.
    """

    # Supplied by the host calculator (declared for the type checker only).
    c: int
    s_max: int
    theta: float
    policy: str
    calc_params: Any
    v: list[float] | None
    w: list[float] | None
    _check_if_servers_and_sources_set: Callable[[], None]
    _build_solver: Callable[[], Any]
    get_v: Callable[[], list[float]]

    def _boundary_stock_probs(self, n: int) -> np.ndarray:
        """Stock distribution at boundary level ``n < c``, summed over configurations."""
        raise NotImplementedError

    def _wait_phase_type(self, level_truncation: int | None = None):
        """Phase-type representation ``(alpha, T, p_zero)`` of the waiting time."""
        self._check_if_servers_and_sources_set()
        solver = self._build_solver()
        c = self.c
        n_max = level_truncation or self.calc_params.p_num
        n_max = max(n_max, c)

        repeating: list[np.ndarray] = []
        vec = solver.pi1.copy()
        for _ in range(n_max - c + 1):
            repeating.append(vec)
            vec = vec @ solver.r

        return build_wait_phase_type_blocks(
            boundary_stock_probs=self._boundary_stock_probs,
            repeating_phase_probs=lambda n: repeating[n - c],
            a0=solver.a0,
            a1=solver.a1,
            a2=solver.a2,
            m=self.s_max + 1,
            c=c,
            theta=self.theta,
            n_max=n_max,
            lost_sales=self.policy == "lost_sales",
        )

    def get_w_moments(self, num: int = 4, level_truncation: int | None = None) -> list[float]:
        """
        Exact raw moments of the waiting time, from the tagged-customer
        phase-type representation (see :func:`build_wait_phase_type_blocks`).
        """
        alpha, t_mat, _ = self._wait_phase_type(level_truncation)
        return phase_type_moments(alpha, t_mat, num)

    def get_tail(self, t: float, level_truncation: int | None = None) -> float:
        """Exact P(W > t) -- deadline-violation probability for the wait."""
        alpha, t_mat, _ = self._wait_phase_type(level_truncation)
        return phase_type_tail(alpha, t_mat, t)

    def get_cdf(self, t: float, level_truncation: int | None = None) -> float:
        """Exact P(W <= t)."""
        return 1.0 - self.get_tail(t, level_truncation)

    def get_w(self) -> list[float]:
        """
        Mean waiting time, exact -- the first moment of :meth:`get_w_moments`.

        NOTE (bug fixed in EPIC-075). This used to be ``E[V] - E[S_active]``,
        where ``E[S_active]`` was the mean time a customer is ACTIVELY served.
        That understates the time a customer occupies a server whenever
        ``c > 1`` and stockouts occur: one server can consume the last stock
        unit while another customer is still mid-service, and that service is
        then suspended until a replenishment arrives. The suspension is part of
        the customer's sojourn but not part of its wait, so subtracting only
        the active part overstates ``E[W]``. The single-server models are
        unaffected -- with one server the stock cannot drop while that service
        runs. See :meth:`get_service_time_mean` for the occupancy time.
        """
        self.w = [self.get_w_moments(1)[0]]
        return self.w

    def get_service_time_mean(self) -> float:
        """
        Mean time from becoming SERVEABLE to departure, ``E[S] = E[V] - E[W]``.

        Exceeds the mean ACTIVE service time (``_mean_service_time``) when
        ``c > 1`` and stockouts occur, since a service in progress is suspended
        while stock is out (see :meth:`get_w`).

        For the exponential and H2 variants this is also the mean time the
        customer OCCUPIES a server, because there a customer holding a server at
        zero stock cannot make any progress. The Erlang variant differs: its
        intermediate phase advances are not blocked by a stockout (only the
        final, stock-consuming one is -- see that module's docstring), so a
        customer can already be advancing phases during what is counted here as
        waiting. Its occupancy time is therefore longer than this value. The
        split is a matter of where one draws the line; ``E[V]`` is the same
        either way, and the line drawn here -- the wait ends once a server is
        free AND an item is in stock -- is the one the queueing-inventory
        literature uses, and it is uniform across all seven calculators.
        """
        v = self.v if self.v is not None else self.get_v()
        w = self.w if self.w is not None else self.get_w()
        return v[0] - w[0]
