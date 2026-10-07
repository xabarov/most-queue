"""
M/H2(p1,mu1,mu2)^[a,b]/1 bulk-service queue: general (not exponential)
batch-service time, fitted to an H2 (hyperexponential, 2-phase mixture)
phase-type representation -- the CV>=1 complement to EPIC-035's Erlang
(CV<=1) case. See docs/research/bulk-service-general-h2-2026.md for why H2
was picked as the natural EPIC-035 continuation instead of the classical
embedded-chain/PGF route (Neuts 1975, Chaudhry-Templeton).

Unlike Erlang (a sequential chain of k phases), H2 is a BRANCH: at the
moment a batch starts service, one of two exponential phases is chosen
once (phase 0 w.p. p1, rate mu1; phase 1 w.p. p2=1-p1, rate mu2) and the
batch stays in that phase until it completes -- no mid-service phase
transitions. Every "a batch starts" transition (idle-threshold reached, or
a completing batch immediately starting the next one) therefore SPLITS
into two weighted sub-transitions (rate x p1 into phase 0, rate x p2 into
phase 1), instead of Erlang's single deterministic "always start in phase
0" transition.

State: (i, phase, j) -- i = batch size in service (0 = idle), phase in
{0,1} = which H2 branch the current batch chose at its start (meaningful
only when i>=1), j = number waiting.
p1=1 (H2 collapses to Exp(mu1)) reduces EXACTLY to BulkServiceMM1Calc --
verified to full float64 precision, the primary regression test.

Mean-only (E[N], E[V], E[W] via Little's law), same scope as EPIC-035's
BulkServiceErlangCalc -- exact moments for this phase-type-augmented case
are a reserve item (would need a PASTA-based tagged-customer argument that
also tracks which phase an arrival finds the batch in).

EPIC-042 adds batch-size-dependent parameters: p1/mu1/mu2 may each be a
callable `f(batch_size) -> value`, porting BulkServiceMM1Calc's existing
callable-mu convention. A subtlety absent from the Erlang case (EPIC-042's
Erlang generalization only needed the CURRENT batch's rate): the branch
SPLIT at a "batch starts" transition must use p1 of the NEW batch about to
form, not of the batch that just completed -- e.g. at a completion event
i -> take (take = min(b, j)), the outflow rate uses the completing batch's
own (i, phase) rate, but the p1/p2 weights splitting into the new batch's
two phases must be p1(take), not p1(i). Getting this backwards would
silently use the wrong batch's H2 parameters whenever p1 varies with size.

EPIC-067 generalizes ``get_tail`` to any ``1 <= a <= b``, porting
``BulkServiceMM1Calc``/``BulkServiceErlangCalc``'s idle-refill race-aware
decomposition: when the remainder after the full batches ahead is below the
threshold ``a``, the wait is a RACE between new Poisson(lambda) arrivals and
the LAST branching layer's completion, not a naive sequential
"service phases, then a separate refill phase" split (confirmed wrong for
the exponential/Erlang cases -- see
``most_queue.theory.batch._idle_refill`` and
docs/epics/EPIC-067-bulk-service-idle-refill.md). The race only attaches to
the FINAL layer; earlier layers always have enough ahead to proceed without
a refill check, so only that one layer needs its 2 (branching) states
crossed with the arrival count.

EPIC-068 added Markovian abandonment (rate ``gamma``) at the ``_solve_pi``
(stationary-distribution) level only; a gamma-aware ``get_abandonment_prob``/
``get_w``/``get_tail`` were explicitly out of scope, since
``BulkServiceMM1Calc``/``BulkServiceErlangCalc``'s shared
``most_queue.theory.batch._idle_refill.abandonment_chain`` assumes a single
SEQUENTIAL chain of service phases ahead of the tagged customer, while H2's
"ahead" batches each independently re-choose their own branch (mu1 w.p. p1,
else mu2).

EPIC-071 closes that reserve with a dedicated H2 construction,
``_h2_abandonment_chain`` (below): instead of crossing every sequential
phase with (R, K) like ``abandonment_chain`` does, H2 needs only 3 "segments"
per (R, K) pair -- "first" (the batch the tagged customer actually observed
on arrival, already resolved to ONE definite memoryless phase by PASTA, so
no branch choice left to make) and "ahead" phase 0 / phase 1 (each of the
subsequent, not-yet-formed batches, which DOES re-choose a branch at the
moment it starts) -- rather than Erlang's n_first-then-k_ahead sequential
phase count. Every transition that moves from one batch to the next (first
-> ahead, or ahead -> ahead) SPLITS into the two branch weights (p1_b,
1-p1_b); the within-segment abandonment (R*gamma) and new-arrival (lam)
transitions never change phase, exactly like the sequential case. The
idle-refill (R, K) race carries over unchanged -- it never depended on which
H2 branch a batch chose. p1=1 (H2 collapses to Exp(mu1)) must reduce exactly
to ``BulkServiceMM1Calc``'s gamma>0 construction -- the primary regression
test, alongside gamma=0 collapsing exactly to the EPIC-067 ``get_tail``.
"""

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.params import H2Params
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.batch._idle_refill import race_subgen


def _h2_abandonment_chain(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    rate_obs: float,
    mu1_b: float,
    mu2_b: float,
    p1_b: float,
    j: int,
    a: int,
    b: int,
    lam: float,
    gamma: float,
    start_idle: bool = False,
):
    """
    EPIC-071: H2-branching analogue of
    ``most_queue.theory.batch._idle_refill.abandonment_chain`` (EPIC-068).

    State: (segment in {first, ahead}, phase in {0, 1} -- meaningful only for
    "ahead" -- R = surviving originally-ahead count 0..j, K = new arrivals
    behind the tagged customer accumulated so far, capped at a-1), plus
    idle/refill states (R, K) with no phase, identical to
    ``abandonment_chain``'s idle-refill race. "first" is a single state per
    (R, K): the batch the tagged customer observed on arrival is already
    resolved to one definite memoryless phase (rate ``rate_obs``), so there
    is no branch choice left there. Every service completion that excludes
    the tagged customer (always a batch of size exactly ``b`` -- same proved
    fact as ``abandonment_chain``) starts a NEW "ahead" batch, which
    re-chooses its branch: phase 0 at ``mu1_b`` w.p. ``p1_b``, else phase 1
    at ``mu2_b``. R-abandonment and new-K-arrival transitions never change
    phase or segment.

    :return: ``(subgen, m, start, serve_rate)`` -- same contract as
        ``abandonment_chain``.
    """
    n_rk = (j + 1) * a

    def idx_first(r_val, k_val):
        return r_val * a + k_val

    base_ahead = 0 if start_idle else n_rk

    def idx_ahead(phase, r_val, k_val):
        return base_ahead + phase * n_rk + r_val * a + k_val

    base_idle = base_ahead + (0 if start_idle else 2 * n_rk)

    def idx_idle(r_val, k_val):
        return base_idle + r_val * a + k_val

    m = base_idle + n_rk
    rows, cols, vals = [], [], []
    out_rate = np.zeros(m)
    serve_rate = np.zeros(m)

    def add(src, dst, rate):
        rows.append(src)
        cols.append(dst)
        vals.append(rate)
        out_rate[src] += rate

    def build_segment(idx_fn, rate):
        for r_val in range(j + 1):
            for k_val in range(a):
                s = idx_fn(r_val, k_val)
                total = r_val + 1 + k_val
                if total >= a:
                    take = min(b, total)
                    if take >= r_val + 1:
                        out_rate[s] += rate
                        serve_rate[s] += rate
                    else:  # take == b (proved in abandonment_chain), excludes tagged
                        nr = r_val - take
                        add(s, idx_ahead(0, nr, k_val), rate * p1_b)
                        add(s, idx_ahead(1, nr, k_val), rate * (1.0 - p1_b))
                else:
                    add(s, idx_idle(r_val, k_val), rate)
                if r_val >= 1:
                    add(s, idx_fn(r_val - 1, k_val), r_val * gamma)
                if k_val < a - 1:
                    add(s, idx_fn(r_val, k_val + 1), lam)

    if not start_idle:
        build_segment(idx_first, rate_obs)
        build_segment(lambda r, k: idx_ahead(0, r, k), mu1_b)
        build_segment(lambda r, k: idx_ahead(1, r, k), mu2_b)

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

    # See abandonment_chain's identical comment: idle/refill states are only
    # reachable when the idle-refill regime is actually reachable; unreachable
    # rows get a harmless positive out_rate pin so (-A) stays invertible.
    out_rate[out_rate == 0] = 1.0

    q = sp.coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
    subgen = (q - sp.diags(out_rate)).tocsc()
    start = idx_idle(j, 0) if start_idle else idx_first(j, 0)
    return subgen, m, start, serve_rate


class BulkServiceH2Calc(BaseQueue):
    """
    Exact M/H2(p1,mu1,mu2)^[a,b]/1 bulk-service queue via a truncated,
    phase-augmented CTMC.

    :param a: minimum batch size to start a service.
    :param b: maximum batch size.
    :param queue_truncation: cap on the number waiting (state-space bound).
    """

    def __init__(self, a: int, b: int, queue_truncation: int = 300, gamma: float = 0.0):
        super().__init__(n=1)
        if not 1 <= a <= b:
            raise ValueError("require 1 <= a <= b")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.a = a
        self.b = b
        self.N = queue_truncation
        self.gamma = gamma
        self.l = None
        self.p1_fn = None
        self.mu1_fn = None
        self.mu2_fn = None
        self._pi: np.ndarray | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, p1, mu1, mu2):  # pylint: disable=arguments-differ
        """
        :param p1: probability of choosing phase 0 (rate mu1) at batch start; phase 1 (rate mu2)
            w.p. 1-p1. Each of p1/mu1/mu2 may be a scalar (batch-size-independent) or a callable
            f(batch_size) -> value (e.g. LLM batching where a bigger batch has different
            variability/rate).
        """

        def _as_fn(value):
            if callable(value):
                return value
            return lambda _i, _v=float(value): _v

        self.p1_fn = _as_fn(p1)
        self.mu1_fn = _as_fn(mu1)
        self.mu2_fn = _as_fn(mu2)
        for size in range(self.a, self.b + 1):
            p1_val, mu1_val, mu2_val = self.p1_fn(size), self.mu1_fn(size), self.mu2_fn(size)
            if not 0.0 <= p1_val <= 1.0:
                raise ValueError(f"p1({size}) must be in [0, 1], got {p1_val}")
            if mu1_val <= 0 or mu2_val <= 0:
                raise ValueError(f"mu1({size}), mu2({size}) must be positive, got {mu1_val}, {mu2_val}")
        self.is_servers_set = True

    def set_servers_from_moments(self, moments: list[float]):
        """Fit (p1, mu1, mu2) from raw moments (mean, second moment, ...) via H2Distribution.get_params."""
        params: H2Params = H2Distribution.get_params(moments)
        self.set_servers(params.p1, params.mu1, params.mu2)

    def _idle_index(self, j: int) -> int:
        return j

    def _busy_index(self, i: int, phase: int, j: int) -> int:
        return (self.N + 1) + ((i - 1) * 2 + phase) * (self.N + 1) + j

    def _solve_pi(self) -> np.ndarray:
        """Build and solve the CTMC; cache and return the stationary distribution pi."""
        if self._pi is not None:
            return self._pi
        self._check_if_servers_and_sources_set()

        a, b, N, lam = self.a, self.b, self.N, self.l
        gamma = self.gamma
        p1_fn, mu1_fn, mu2_fn = self.p1_fn, self.mu1_fn, self.mu2_fn
        n_states = (N + 1) + b * 2 * (N + 1)
        rows, cols, vals = [], [], []

        def add(src, dst, r):
            rows.append(src)
            cols.append(dst)
            vals.append(r)

        p1_a = p1_fn(a)
        for j in range(N + 1):
            s = self._idle_index(j)
            if j + 1 < a:
                add(s, self._idle_index(j + 1), lam)
            else:
                add(s, self._busy_index(a, 0, 0), lam * p1_a)
                add(s, self._busy_index(a, 1, 0), lam * (1.0 - p1_a))
            if j >= 1 and gamma > 0:
                add(s, self._idle_index(j - 1), j * gamma)

        for i in range(1, b + 1):
            for phase, mu in ((0, mu1_fn(i)), (1, mu2_fn(i))):
                for j in range(N + 1):
                    s = self._busy_index(i, phase, j)
                    if j < N:
                        add(s, self._busy_index(i, phase, j + 1), lam)
                    if j >= a:
                        take = min(b, j)
                        p1_take = p1_fn(take)
                        add(s, self._busy_index(take, 0, j - take), mu * p1_take)
                        add(s, self._busy_index(take, 1, j - take), mu * (1.0 - p1_take))
                    else:
                        add(s, self._idle_index(j), mu)
                    if j >= 1 and gamma > 0:
                        add(s, self._busy_index(i, phase, j - 1), j * gamma)

        q = sp.coo_matrix((vals, (rows, cols)), shape=(n_states, n_states)).tocsr()
        out = np.asarray(q.sum(axis=1)).ravel()
        q = q - sp.diags(out)

        mat = q.transpose().tolil()
        mat[0, :] = 1.0
        rhs = np.zeros(n_states)
        rhs[0] = 1.0
        pi = spla.spsolve(mat.tocsr(), rhs)
        pi = np.maximum(np.real(pi), 0.0)
        pi = pi / pi.sum()
        self._pi = pi
        return pi

    def _abandon_chain_for_state(self, rate_obs: float, j: int, start_idle: bool = False):
        """EPIC-071: build the H2 abandonment chain for a given observed state."""
        b, lam, gamma = self.b, self.l, self.gamma
        mu1_b, mu2_b, p1_b = self.mu1_fn(b), self.mu2_fn(b), self.p1_fn(b)
        return _h2_abandonment_chain(rate_obs, mu1_b, mu2_b, p1_b, j, self.a, b, lam, gamma, start_idle=start_idle)

    def get_abandonment_prob(self) -> float:
        """
        EPIC-071: exact probability that a PASTA-arriving (tagged) customer
        abandons (Markovian patience, rate ``gamma``) before their own batch
        starts service. 0.0 when ``gamma == 0``. See
        ``BulkServiceErlangCalc.get_abandonment_prob``/module docstring for
        the H2-branching generalization of the EPIC-068 construction.
        """
        if self.gamma <= 0:
            return 0.0
        pi = self._solve_pi()
        a, b = self.a, self.b
        cache: dict[tuple, float] = {}
        p_abandon = 0.0
        for j in range(self.N + 1):
            prob = pi[self._idle_index(j)]
            if prob <= 0 or j == a - 1:
                continue
            key = ("idle", j)
            cached = cache.get(key)
            if cached is None:
                subgen, m, start, _ = self._abandon_chain_for_state(0.0, j, start_idle=True)
                alpha = np.zeros(m)
                alpha[start] = 1.0
                x = spla.spsolve((-subgen).tocsc(), np.ones(m))
                cached = self.gamma * float(alpha @ x)
                cache[key] = cached
            p_abandon += prob * cached

        for i in range(1, b + 1):
            for phase, rate_obs in ((0, self.mu1_fn(i)), (1, self.mu2_fn(i))):
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, phase, j)]
                    if prob <= 0:
                        continue
                    key = ("busy", rate_obs, j)
                    cached = cache.get(key)
                    if cached is None:
                        subgen, m, start, _ = self._abandon_chain_for_state(rate_obs, j)
                        alpha = np.zeros(m)
                        alpha[start] = 1.0
                        x = spla.spsolve((-subgen).tocsc(), np.ones(m))
                        cached = self.gamma * float(alpha @ x)
                        cache[key] = cached
                    p_abandon += prob * cached
        return p_abandon

    def get_w(self, num: int = 4) -> list[float]:
        """
        Exact raw moments of W conditional on being served, ``gamma > 0``
        (EPIC-071). At ``gamma == 0`` exact moments beyond the mean (via
        ``run()``'s Little's-law ``E[W]``) remain a separate, unimplemented
        reserve item -- see the module docstring.
        """
        if self.gamma <= 0:
            raise NotImplementedError(
                "get_w() is only implemented for gamma > 0 (EPIC-071 Markovian abandonment). "
                "Exact W moments at gamma == 0 beyond the mean (via run()) remain a separate, "
                "unimplemented reserve item -- see the module docstring."
            )
        return self._get_w_with_abandonment(num)

    def _get_w_with_abandonment(self, num: int = 4) -> list[float]:
        """EPIC-071: exact raw moments of W conditional on being served, gamma > 0."""
        pi = self._solve_pi()
        a, b = self.a, self.b
        cache: dict[tuple, tuple] = {}
        numerator = np.zeros(num)
        p_served_total = 0.0

        def accumulate(prob, key, builder):
            nonlocal p_served_total, numerator
            cached = cache.get(key)
            if cached is None:
                subgen, m, start, serve_rate = builder()
                alpha = np.zeros(m)
                alpha[start] = 1.0
                neg_a = (-subgen).tocsc()
                x = spla.spsolve(neg_a, serve_rate)
                moments = []
                for n in range(1, num + 1):
                    x = spla.spsolve(neg_a, x)
                    moments.append(math.factorial(n) * float(alpha @ x))
                p_served_state = 1.0 - self.gamma * float(alpha @ spla.spsolve(neg_a, np.ones(m)))
                cached = (moments, p_served_state)
                cache[key] = cached
            moments, p_served_state = cached
            numerator += prob * np.array(moments)
            p_served_total += prob * p_served_state

        for j in range(self.N + 1):
            prob = pi[self._idle_index(j)]
            if prob <= 0:
                continue
            if j == a - 1:
                p_served_total += prob
                continue
            accumulate(prob, ("idle", j), lambda j=j: self._abandon_chain_for_state(0.0, j, start_idle=True))

        for i in range(1, b + 1):
            for phase, rate_obs in ((0, self.mu1_fn(i)), (1, self.mu2_fn(i))):
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, phase, j)]
                    if prob <= 0:
                        continue
                    accumulate(
                        prob,
                        ("busy", rate_obs, j),
                        lambda rate_obs=rate_obs, j=j: self._abandon_chain_for_state(rate_obs, j),
                    )
        if p_served_total <= 0:
            return [0.0] * num
        return list(numerator / p_served_total)

    def get_tail(self, t: float) -> float:
        """
        Exact P(W > t), any ``1 <= a <= b``.

        PASTA: a tagged arrival sees stationary (i, phase, j) [or, if idle,
        (0, j)]. If busy (i>=1), its wait decomposes into the OBSERVED
        current batch's remaining time -- Exp(mu[phase](i)), memoryless,
        same phase it was already in -- followed by ``j // b`` FULL batches
        ahead, each independently redrawing its OWN H2 phase (rate mu1(b)
        w.p. p1(b), else mu2(b)) once it starts. Unlike Erlang's purely
        sequential chain, this "ahead" part genuinely BRANCHES at every
        subsequent batch boundary.

        EPIC-067 adds the idle-refill case: if the remainder after the full
        batches ahead (plus the tagged customer) is below the threshold
        ``a``, the wait is a RACE between new Poisson(lambda) arrivals and
        the LAST branching layer's completion (the earlier layers always
        have enough ahead to proceed without a refill check, same argument
        as ``BulkServiceMM1Calc``/``BulkServiceErlangCalc`` -- see
        ``most_queue.theory.batch._idle_refill`` and
        docs/epics/EPIC-067-bulk-service-idle-refill.md). The branching only
        matters for the layer the race attaches to; earlier layers and the
        idle-state refill are non-branching (the arrival-counting race
        doesn't care which H2 phase an EARLIER, already-resolved batch was
        in). p1=1 (H2 collapses to Exp(mu1)) reduces exactly to
        ``BulkServiceMM1Calc.get_tail``.

        At ``gamma > 0`` (EPIC-071), returns the conditional
        ``P(W > t AND served) / P(served)`` instead -- see
        ``_get_tail_with_abandonment``.
        """
        if t < 0:
            raise ValueError("t must be nonnegative")
        if self.gamma > 0:
            return self._get_tail_with_abandonment(t)
        pi = self._solve_pi()
        a, b = self.a, self.b
        lam = self.l
        ahead_params = (self.mu1_fn(b), self.mu2_fn(b), self.p1_fn(b))
        cache: dict[tuple, float] = {}

        tail = 0.0
        for j in range(self.N + 1):
            prob = pi[self._idle_index(j)]
            need = a - j - 1
            if prob > 0 and need > 0:
                key = ("idle", need)
                cached = cache.get(key)
                if cached is None:
                    # pure Poisson(lam) refill, no service phase: a trivial 1-rate-repeated chain
                    rates = np.full(need, lam)
                    subgen = sp.diags([-rates, rates[:-1]], [0, 1], format="csc")
                    cached = float(spla.expm_multiply(subgen * t, np.ones(need))[0])
                    cache[key] = cached
                tail += prob * cached

        for i in range(1, b + 1):
            for phase, rate_obs in ((0, self.mu1_fn(i)), (1, self.mu2_fn(i))):
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, phase, j)]
                    if prob <= 0:
                        continue
                    full_ahead = j // b
                    remainder = j - full_ahead * b
                    need = a - remainder - 1
                    key = (rate_obs, full_ahead, max(need, 0))
                    cached = cache.get(key)
                    if cached is None:
                        if need > 0:
                            cached = self._tagged_wait_tail_with_refill(
                                rate_obs, full_ahead, ahead_params, lam, need, t
                            )
                        else:
                            cached = self._tagged_wait_tail(rate_obs, full_ahead, ahead_params, t)
                        cache[key] = cached
                    tail += prob * cached
        return tail

    @staticmethod
    def _tagged_wait_tail(rate_obs, full_ahead, ahead_params, t) -> float:
        """Tail of [Exp(rate_obs)] + [full_ahead independent fresh-phase H2(b) batches]."""
        mu1_b, mu2_b, p1_b = ahead_params
        m = 1 + 2 * full_ahead
        rows, cols, vals = [0], [0], [-rate_obs]
        if full_ahead > 0:
            rows += [0, 0]
            cols += [1, 2]
            vals += [rate_obs * p1_b, rate_obs * (1.0 - p1_b)]
        for k in range(1, full_ahead + 1):
            idx0, idx1 = 1 + (k - 1) * 2, 1 + (k - 1) * 2 + 1
            rows += [idx0, idx1]
            cols += [idx0, idx1]
            vals += [-mu1_b, -mu2_b]
            if k < full_ahead:
                nxt0, nxt1 = 1 + k * 2, 1 + k * 2 + 1
                rows += [idx0, idx0, idx1, idx1]
                cols += [nxt0, nxt1, nxt0, nxt1]
                vals += [mu1_b * p1_b, mu1_b * (1.0 - p1_b), mu2_b * p1_b, mu2_b * (1.0 - p1_b)]
        subgen = sp.coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsc()
        return float(spla.expm_multiply(subgen * t, np.ones(m))[0])

    @staticmethod
    def _tagged_wait_tail_with_refill(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        rate_obs, full_ahead, ahead_params, lam, need, t
    ) -> float:
        """Tail of [Exp(rate_obs)] + [full_ahead H2(b) batches] + [idle-refill race], need > 0.

        New arrivals can occur during ANY of the (1 + full_ahead) service
        phases ahead of the tagged customer -- not just the last one -- so
        the arrival-count dimension (0..need) is crossed with EVERY layer
        (layer 0: 1 phase-state; layers 1..full_ahead: 2 branching
        phase-states each), exactly like ``_idle_refill.race_subgen``'s
        sequential-chain construction, just with a per-layer branching
        factor. An earlier version only crossed the LAST layer with the
        count dimension (silently dropping arrivals during earlier phases)
        -- confirmed wrong against an independent simulation (~20% error on
        specific states) before this fix; see
        docs/epics/EPIC-067-bulk-service-idle-refill.md.
        """
        mu1_b, mu2_b, p1_b = ahead_params
        if full_ahead == 0:
            subgen, m, start = race_subgen(np.array([rate_obs]), lam, need)
            return float(spla.expm_multiply(subgen * t, np.ones(m))[start])

        width = need + 1
        n_grid = width + full_ahead * 2 * width  # layer 0 (1 phase) + layers 1..full_ahead (2 phases)
        m = n_grid + need  # + refill tail

        def idx(layer, phase, count):  # layer 0 ignores phase (always 0)
            if layer == 0:
                return count
            return width + (layer - 1) * 2 * width + phase * width + count

        def refill_idx(remaining):  # remaining in 1..need
            return n_grid + (remaining - 1)

        rows, cols, vals = [], [], []
        out_rate = np.zeros(m)

        def add(s, d, r):
            rows.append(s)
            cols.append(d)
            vals.append(r)
            out_rate[s] += r

        for count in range(width):
            s = idx(0, 0, count)
            if count < need:
                add(s, idx(0, 0, count + 1), lam)
            if full_ahead == 0:
                if count == need:
                    out_rate[s] += rate_obs
                else:
                    add(s, refill_idx(need - count), rate_obs)
            else:
                add(s, idx(1, 0, count), rate_obs * p1_b)
                add(s, idx(1, 1, count), rate_obs * (1.0 - p1_b))

        for layer in range(1, full_ahead + 1):
            for phase, r in ((0, mu1_b), (1, mu2_b)):
                for count in range(width):
                    s = idx(layer, phase, count)
                    if count < need:
                        add(s, idx(layer, phase, count + 1), lam)
                    if layer == full_ahead:
                        if count == need:
                            out_rate[s] += r  # enough already accumulated -> absorbs directly
                        else:
                            add(s, refill_idx(need - count), r)
                    else:
                        add(s, idx(layer + 1, 0, count), r * p1_b)
                        add(s, idx(layer + 1, 1, count), r * (1.0 - p1_b))

        for remaining in range(1, need + 1):
            s = refill_idx(remaining)
            if remaining > 1:
                add(s, refill_idx(remaining - 1), lam)
            out_rate[s] += lam  # remaining==1 absorbs directly

        q = sp.coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
        subgen = (q - sp.diags(out_rate)).tocsc()
        return float(spla.expm_multiply(subgen * t, np.ones(m))[0])

    def _get_tail_with_abandonment(self, t: float) -> float:
        """EPIC-071: exact P(W > t AND served) / P(served), gamma > 0."""
        pi = self._solve_pi()
        a, b = self.a, self.b
        cache: dict[tuple, tuple] = {}
        numerator = 0.0
        p_served_total = 0.0

        def accumulate(prob, key, builder):
            nonlocal p_served_total, numerator
            cached = cache.get(key)
            if cached is None:
                subgen, m, start, serve_rate = builder()
                alpha = np.zeros(m)
                alpha[start] = 1.0
                neg_a = (-subgen).tocsc()
                v = spla.spsolve(neg_a, serve_rate)
                p_served_state = 1.0 - self.gamma * float(alpha @ spla.spsolve(neg_a, np.ones(m)))
                cached = (subgen, v, start, p_served_state)
                cache[key] = cached
            subgen, v, start, p_served_state = cached
            numerator += prob * float(spla.expm_multiply(subgen * t, v)[start])
            p_served_total += prob * p_served_state

        for j in range(self.N + 1):
            prob = pi[self._idle_index(j)]
            if prob <= 0:
                continue
            if j == a - 1:
                p_served_total += prob
                continue
            accumulate(prob, ("idle", j), lambda j=j: self._abandon_chain_for_state(0.0, j, start_idle=True))

        for i in range(1, b + 1):
            for phase, rate_obs in ((0, self.mu1_fn(i)), (1, self.mu2_fn(i))):
                for j in range(self.N + 1):
                    prob = pi[self._busy_index(i, phase, j)]
                    if prob <= 0:
                        continue
                    accumulate(
                        prob,
                        ("busy", rate_obs, j),
                        lambda rate_obs=rate_obs, j=j: self._abandon_chain_for_state(rate_obs, j),
                    )
        if p_served_total <= 0:
            return 0.0
        return numerator / p_served_total

    def get_cdf(self, t: float) -> float:
        """Exact P(W <= t) -- see ``get_tail`` for scope (a=1 only)."""
        return 1.0 - self.get_tail(t)

    def run(self) -> QueueResults:
        """Solve the CTMC; return mean waiting/sojourn moments (means)."""
        start = self._measure_time()
        pi = self._solve_pi()
        b, N, lam = self.b, self.N, self.l
        p1_fn, mu1_fn, mu2_fn = self.p1_fn, self.mu1_fn, self.mu2_fn

        e_n = 0.0
        p_busy_size = np.zeros(b + 1)  # P(batch size = i | server busy), i=1..b
        for j in range(N + 1):
            e_n += pi[self._idle_index(j)] * j
        for i in range(1, b + 1):
            for phase in (0, 1):
                for j in range(N + 1):
                    prob = pi[self._busy_index(i, phase, j)]
                    e_n += prob * (i + j)
                    p_busy_size[i] += prob

        e_t = e_n / lam

        if self.gamma > 0:
            # EPIC-071: Little's-law e_t counts abandoning customers too (same
            # convention as BulkServiceMM1Calc/BulkServiceErlangCalc at gamma>0);
            # w switches to the exact conditional-on-served value instead.
            e_w = self.get_w(num=1)[0]
        else:

            def mean_service_of(size):
                p1_val = p1_fn(size)
                return p1_val / mu1_fn(size) + (1.0 - p1_val) / mu2_fn(size)

            p_busy_total = p_busy_size.sum()
            if p_busy_total > 0:
                mean_batch_service = sum(mean_service_of(i) * p_busy_size[i] for i in range(1, b + 1)) / p_busy_total
            else:
                mean_batch_service = mean_service_of(b)
            e_w = e_t - mean_batch_service

        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=[e_w, 0, 0, 0],
            utilization=1.0 - float(sum(pi[self._idle_index(j)] for j in range(N + 1))),
        )
        self._set_duration(res, start)
        return res
