"""
M/M^[a,b]/c bulk-service queue: `c` IDENTICAL independent servers sharing one FCFS
queue, each serving batches the same way `BulkServiceMM1Calc` does (starts a batch
once >= `a` are waiting, takes up to `b`). Direct model for a multi-GPU LLM-serving
pool: several continuous-batching replicas behind one shared request router.

EPIC-069 (roadmap direction A2, closing the final batch-service-SLA reserve item
still in scope for the current article -- see
docs/epics/EPIC-069-bulk-service-multiserver.md).

**Why this isn't a direct port of EPIC-033/038/041's state-splitting:** those epics
have heterogeneous servers, but each one still serves ONE customer at a time --
splitting tracks "which specific server holds which job". Here servers are
IDENTICAL, but each holds an entire BATCH of variable size -- what matters is only
HOW MANY servers are currently busy with each batch size, not which physical server.
State: an occupancy vector `(n_1, ..., n_b)` (`n_s` = number of servers currently
serving a batch of size `s`, `sum(n_s) <= c`; `idle_count = c - sum(n_s)`), crossed
with `j` = number waiting in the shared queue.

**Dispatch rule (fixes the ambiguity flagged in the EPIC-069 proposal):** whenever at
least one server is idle, a batch is dispatched the INSTANT the queue reaches `a`
(same convention as `BulkServiceMM1Calc`'s own idle state) -- since arrivals occur
one at a time, an idle-triggered dispatch always takes EXACTLY `a` (never more: you
cannot overshoot a threshold reached one arrival at a time). This means the state
space has a built-in invariant: `idle_count >= 1` implies `j < a` (if the queue
already had `>= a` waiting, some idle server would already have taken them) --
multiple servers can sit idle together, each waiting on the same shared queue to
reach `a`. Only when ALL `c` servers are busy can `j` build up past `a`; the batch a
newly-freed server picks up can then be as large as `min(b, j)` (the queue had the
chance to grow past `a` while every server was occupied) -- exactly like
`BulkServiceMM1Calc`'s own idle-vs-busy split, just generalized to "any server free"
instead of "the one server free".

Since servers are plain exponential (memoryless), a busy server's remaining service
time needs no phase tracking regardless of how long it has already run -- this keeps
the FIRST version of this model (this file) tractable: no Erlang/H2 phase dimension
is crossed with the occupancy vector. EPIC-069's own proposal explicitly scoped
Erlang/H2 batch-service-time generalizations to a later, separate step (if/when
warranted) -- deferred, not silently dropped.

`_solve_pi()`/`get_n_moments()` are exact for ANY `a<=b`, `c>=1`, including
batch-size-dependent `mu(size)`.

`get_w()`/`get_tail()` are exact ONLY for batch-size-INDEPENDENT `mu` (a key
simplification found during derivation, not assumed upfront): when `mu` doesn't
depend on batch size, the aggregate "next completion" rate among all `c` busy
servers is always exactly `c * mu`, REGARDLESS of which specific sizes they are
currently serving (memoryless exponential servers + idle_count staying 0 for as
long as a freed server immediately re-dispatches) -- so a tagged customer's wait,
while all `c` servers are busy, needs no occupancy-vector state at all: the number
of customers ahead of tagged (`R`) decreases by `b` at each at-rate-`c*mu`
"exclusion" event, exactly `most_queue.theory.batch._idle_refill.abandonment_chain`'s
own "ahead" segment with the competing-abandonment rate set to 0 (no abandonment in
this model) and a single rate throughout (no "first vs ahead" distinction needed,
since there is no "observed batch, partway through" concept when any of `c`
identical memoryless servers could be the one that frees next) -- reused as-is, not
reimplemented. This does NOT generalize to batch-size-dependent `mu`: there, the
aggregate rate genuinely depends on the full occupancy vector, which must then be
tracked inside the absorbing chain -- a real state-space blowup, deferred as a
documented reserve (`_require_constant_mu`), not silently dropped.

EPIC-070 adds Markovian abandonment (rate `gamma`, `MM1Impatience` convention) on
top of the batch-size-independent case -- the first of three "combination" reserve
items flagged after EPIC-068/069. This combination turned out to need NO new
construction: the `c*mu`-aggregate-rate simplification above already reduces the
busy-state tagged-wait problem to `abandonment_chain`'s own "ahead" segment, and
that function already supports `gamma>0` natively (EPIC-068 built it with a
competing patience exit from the start) -- passing the real `gamma` through
instead of fixing it at 0.0 is the entire extension. `_solve_pi()` gets the same
`j*gamma` reneging term `BulkServiceMM1Calc`/`BulkServiceErlangCalc` already have.

EPIC-072 closes the batch-size-DEPENDENT `mu(size)` reserve (`gamma == 0` only --
combining it with EPIC-070's abandonment stays a separate, unimplemented reserve,
see `_require_constant_mu`'s message): `_occupancy_busy_chain` (below) tracks the
occupancy vector as EXPLICIT tagged-wait state, but a key simplification (found
during derivation, not assumed) keeps the state space far smaller than the
"full combinatorial blowup" the EPIC-072 proposal doc worried about -- occupancy
tracking is only needed while ALL `c` servers stay busy. The moment a freed
server's slot finds fewer than `a` customers available (so it CAN'T immediately
redispatch and goes idle), occupancy becomes irrelevant to the tagged wait: any
further completion among the REMAINING busy servers cannot change how many
customers are waiting (a completing server with an unchanged queue total just
creates another idle slot, not a dispatch), so from that point on the dynamics
are the exact same occupancy-free `(R, K)` Poisson-refill race already used by
`most_queue.theory.batch._idle_refill.abandonment_chain`'s idle substate (reused
here verbatim, gamma fixed at 0). This cuts the tracked occupancy space down to
vectors with `sum(occ) == c` exactly (not `<= c`, the full set `_solve_pi()`
needs) -- noticeably smaller, and bounded independent of the queue truncation `N`.
"""

import math
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.batch._idle_refill import abandonment_chain
from most_queue.theory.utils.conv import conv_moments


def _enumerate_occupancy(b: int, c: int) -> list[tuple[int, ...]]:
    """All occupancy vectors (n_1, ..., n_b) with each n_s >= 0 and sum(n_s) <= c."""
    results: list[tuple[int, ...]] = []

    def rec(pos: int, remaining: int, cur: list[int]):
        if pos == b:
            results.append(tuple(cur))
            return
        for v in range(remaining + 1):
            cur.append(v)
            rec(pos + 1, remaining - v, cur)
            cur.pop()

    rec(0, c, [])
    return results


def _occupancy_busy_chain(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    occ_full_list: list[tuple[int, ...]],
    occ_full_index: dict[tuple[int, ...], int],
    mu_fn: Callable[[int], float],
    start_occ_idx: int,
    j: int,
    a: int,
    b: int,
    lam: float,
):
    """
    EPIC-072: absorbing chain for a tagged customer's wait, batch-size-DEPENDENT
    `mu`, `gamma == 0` (no abandonment -- that combination stays a separate
    reserve). `occ_full_list` must contain only occupancy vectors with
    `sum(occ) == c` (the "every server busy" subspace) -- see the module
    docstring for why the idle-refill tail doesn't need occupancy at all, and so
    is folded into an occupancy-FREE `(R, K)` sub-chain exactly like
    `most_queue.theory.batch._idle_refill.abandonment_chain`'s own idle substate
    (gamma fixed at 0 here, so no R-abandonment transitions).

    State: (occupancy index into `occ_full_list`, R = surviving originally-ahead
    count 0..j, K = new arrivals behind the tagged customer, capped at a-1) while
    busy, plus occupancy-free idle states (R, K). Each occupied size class `sz`
    (`occ[sz-1] > 0`) contributes its OWN competing completion rate
    `occ[sz-1] * mu_fn(sz)` -- multiple simultaneous races per state, unlike the
    constant-mu case's single aggregate rate. A completion that excludes the
    tagged customer (`take < r+1`) deterministically updates the occupancy
    (`occ[sz-1] -= 1`, `occ[take-1] += 1`, since the freed slot always
    immediately redispatches when `total >= a`) and resumes the busy chain at
    `R - take`; a completion where `total < a` instead exits into the
    occupancy-free idle substate.

    :return: ``(subgen, m, start, serve_rate)`` -- same contract as
        ``abandonment_chain``.
    """
    n_occ = len(occ_full_list)
    n_rk = (j + 1) * a

    def idx_busy(occ_idx, r, k):
        return occ_idx * n_rk + r * a + k

    base_idle = n_occ * n_rk

    def idx_idle(r, k):
        return base_idle + r * a + k

    m = base_idle + n_rk
    rows, cols, vals = [], [], []
    out_rate = np.zeros(m)
    serve_rate = np.zeros(m)

    def add(src, dst, rate):
        rows.append(src)
        cols.append(dst)
        vals.append(rate)
        out_rate[src] += rate

    for occ_idx, occ in enumerate(occ_full_list):  # pylint: disable=too-many-nested-blocks
        active = [(sz, occ[sz - 1] * mu_fn(sz)) for sz in range(1, b + 1) if occ[sz - 1] > 0]
        for r in range(j + 1):
            for k in range(a):
                s = idx_busy(occ_idx, r, k)
                total = r + 1 + k
                for sz, rate_sz in active:
                    if total >= a:
                        take = min(b, total)
                        if take >= r + 1:
                            out_rate[s] += rate_sz
                            serve_rate[s] += rate_sz
                        else:
                            new_occ = list(occ)
                            new_occ[sz - 1] -= 1
                            new_occ[take - 1] += 1
                            new_idx = occ_full_index[tuple(new_occ)]
                            add(s, idx_busy(new_idx, r - take, k), rate_sz)
                    else:
                        add(s, idx_idle(r, k), rate_sz)
                if k < a - 1:
                    add(s, idx_busy(occ_idx, r, k + 1), lam)

    for r in range(j + 1):
        for k in range(a):
            s = idx_idle(r, k)
            total = r + 1 + k
            if total < a and k < a - 1:
                if r + 1 + (k + 1) >= a:
                    out_rate[s] += lam
                    serve_rate[s] += lam
                else:
                    add(s, idx_idle(r, k + 1), lam)

    # Unreachable rows (e.g. idle states when a=1, never entered) get a harmless
    # positive out_rate pin -- see abandonment_chain's identical comment.
    out_rate[out_rate == 0] = 1.0

    q = sp.coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
    subgen = (q - sp.diags(out_rate)).tocsc()
    start = idx_busy(start_occ_idx, j, 0)
    return subgen, m, start, serve_rate


class BulkServiceMultiserverCalc(BaseQueue):
    """
    Exact M/M^[a,b]/c bulk-service queue (c identical servers, shared FCFS queue)
    via a truncated CTMC on (occupancy vector, number waiting).

    :param a: minimum batch size to start a service.
    :param b: maximum batch size.
    :param c: number of identical parallel servers.
    :param queue_truncation: cap on the number waiting when all `c` servers are busy.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, a: int = 1, b: int = 1, c: int = 1, queue_truncation: int = 300, gamma: float = 0.0
    ):
        super().__init__(n=c)
        if not 1 <= a <= b:
            raise ValueError("require 1 <= a <= b")
        if c < 1:
            raise ValueError(f"c must be >= 1, got {c}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.a = a
        self.b = b
        self.c = c
        self.N = queue_truncation
        self.gamma = gamma
        self.l = None
        self.mu_fn: Callable[[int], float] | None = None
        self._is_constant_mu: bool = False
        self._pi: np.ndarray | None = None
        self._occ_list: list[tuple[int, ...]] | None = None
        self._occ_index: dict[tuple[int, ...], int] | None = None
        self._occ_full_list: list[tuple[int, ...]] | None = None
        self._occ_full_index: dict[tuple[int, ...], int] | None = None
        self._states: list[tuple[int, int]] | None = None
        self._state_index: dict[tuple[int, int], int] | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu):  # pylint: disable=arguments-differ
        """
        :param mu: per-server batch-service rate. A scalar (batch-size-independent,
            Exp(mu)), or a callable mu(batch_size) -> rate.
        """
        if callable(mu):
            self.mu_fn = mu
            self._is_constant_mu = False
        else:
            self.mu_fn = lambda _i, _mu=float(mu): _mu
            self._is_constant_mu = True
        self.is_servers_set = True

    def _build_state_space(self):
        occ_list = _enumerate_occupancy(self.b, self.c)
        occ_index = {occ: i for i, occ in enumerate(occ_list)}
        states: list[tuple[int, int]] = []
        state_index: dict[tuple[int, int], int] = {}
        for oi, occ in enumerate(occ_list):
            idle = self.c - sum(occ)
            max_j = (self.a - 1) if idle >= 1 else self.N
            for j in range(max_j + 1):
                state_index[(oi, j)] = len(states)
                states.append((oi, j))
        self._occ_list = occ_list
        self._occ_index = occ_index
        self._states = states
        self._state_index = state_index

    def _solve_pi(self) -> np.ndarray:  # pylint: disable=too-many-locals
        """Build and solve the CTMC; cache and return the stationary distribution pi."""
        if self._pi is not None:
            return self._pi
        self._check_if_servers_and_sources_set()

        a, b, N, lam, mu_fn = self.a, self.b, self.N, self.l, self.mu_fn
        self._build_state_space()
        occ_list, occ_index, states, state_index = (
            self._occ_list,
            self._occ_index,
            self._states,
            self._state_index,
        )
        n_states = len(states)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        def add(src, dst, rate):
            rows.append(src)
            cols.append(dst)
            vals.append(rate)

        for s_idx, (oi, j) in enumerate(states):
            occ = occ_list[oi]
            idle = self.c - sum(occ)
            # arrival
            if idle >= 1:
                if j + 1 < a:
                    add(s_idx, state_index[(oi, j + 1)], lam)
                else:  # threshold reached -> dispatch exactly `a` to one idle server
                    new_occ = list(occ)
                    new_occ[a - 1] += 1
                    dst_oi = occ_index[tuple(new_occ)]
                    add(s_idx, state_index[(dst_oi, 0)], lam)
            elif j < N:
                add(s_idx, state_index[(oi, j + 1)], lam)
            # service completions (one per currently-busy batch size)
            for sz in range(1, b + 1):
                n_sz = occ[sz - 1]
                if n_sz == 0:
                    continue
                rate = n_sz * mu_fn(sz)
                new_occ = list(occ)
                new_occ[sz - 1] -= 1
                if j >= a:  # only possible when idle == 0 (else j < a by construction)
                    take = min(b, j)
                    new_occ[take - 1] += 1
                    new_j = j - take
                else:
                    new_j = j
                dst_oi = occ_index[tuple(new_occ)]
                add(s_idx, state_index[(dst_oi, new_j)], rate)
            # abandonment: each of the j waiting customers independently reneges
            if j >= 1 and self.gamma > 0:
                add(s_idx, state_index[(oi, j - 1)], j * self.gamma)

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

    def get_n_moments(self, num: int = 4) -> list[float]:
        """Exact raw moments of N (number in system), any a, b, c -- direct summation over pi."""
        pi = self._solve_pi()
        occ_list, states = self._occ_list, self._states
        n = np.array([j + sum(sz * occ_list[oi][sz - 1] for sz in range(1, self.b + 1)) for oi, j in states])
        return [float((pi * n**k).sum()) for k in range(1, num + 1)]

    def _require_constant_mu(self, method_name: str):
        if not self._is_constant_mu:
            raise NotImplementedError(
                f"{method_name} is only implemented for batch-size-INDEPENDENT mu "
                "(a single server's completion rate the same for every batch size). "
                "Combining batch-size-dependent mu with gamma > 0 abandonment needs the "
                "occupancy-vector tracked INSIDE the abandonment race too (EPIC-072's "
                "_occupancy_busy_chain has no R-abandonment transitions) -- a documented "
                "reserve, see the EPIC-072 doc. (gamma == 0: see get_w()/get_tail(), which "
                "DO support batch-size-dependent mu.)"
            )

    def _ensure_full_occ_index(self):
        """EPIC-072: lazily build the "every server busy" occupancy subspace
        (`sum(occ) == c`), the only one the batch-size-dependent tagged-wait
        construction needs -- see module docstring."""
        if self._occ_full_list is None:
            self._occ_full_list = [occ for occ in self._occ_list if sum(occ) == self.c]
            self._occ_full_index = {occ: i for i, occ in enumerate(self._occ_full_list)}

    def _busy_state_moments(self, j: int, num: int, cache: dict) -> list[float]:
        """Exact raw moments of W (restricted to the all-c-servers-busy case), observed R=j
        ahead-of-tagged customers. Since mu is batch-size-independent, the NEXT completion
        among all c busy servers occurs at the constant aggregate rate c*mu REGARDLESS of
        which specific sizes they are serving (memoryless, and idle_count stays 0 for as
        long as a freed server immediately re-dispatches) -- so, unlike the single-server
        EPIC-067 race (which needed a deterministic 'first batch, then full-ahead batches'
        phase sequence), there is no phase sequence to track at all: R decreases by `b`
        at each at-rate-c*mu completion event that excludes the tagged customer, exactly
        the EPIC-068 abandonment_chain 'ahead' segment with gamma=0 (no competing exit) and
        a single rate throughout -- reused as-is, not reimplemented."""
        cached = cache.get(j)
        if cached is not None:
            return cached
        rate = self.c * self.mu_fn(1)
        subgen, m, start, _ = abandonment_chain(rate, 1, rate, 1, j, self.a, self.b, self.l, 0.0, start_idle=False)
        alpha = np.zeros(m)
        alpha[start] = 1.0
        neg_a = (-subgen).tocsc()
        x = np.ones(m)
        moments = []
        for n in range(1, num + 1):
            x = spla.spsolve(neg_a, x)
            moments.append(math.factorial(n) * float(alpha @ x))
        cache[j] = moments
        return moments

    def _busy_state_moments_dependent_mu(self, occ_idx_full: int, j: int, num: int, cache: dict) -> list[float]:
        """EPIC-072: exact raw moments of W (all-c-busy case), batch-size-DEPENDENT mu,
        gamma == 0. Same standard phase-type raw-moment formula as `_busy_state_moments`
        (every absorption is a "served" absorption when gamma == 0, no reward-vector
        restriction needed) -- only the underlying chain differs, via
        `_occupancy_busy_chain` instead of the single-rate `abandonment_chain`."""
        key = (occ_idx_full, j)
        cached = cache.get(key)
        if cached is not None:
            return cached
        self._ensure_full_occ_index()
        subgen, m, start, _ = _occupancy_busy_chain(
            self._occ_full_list, self._occ_full_index, self.mu_fn, occ_idx_full, j, self.a, self.b, self.l
        )
        alpha = np.zeros(m)
        alpha[start] = 1.0
        neg_a = (-subgen).tocsc()
        x = np.ones(m)
        moments = []
        for n in range(1, num + 1):
            x = spla.spsolve(neg_a, x)
            moments.append(math.factorial(n) * float(alpha @ x))
        cache[key] = moments
        return moments

    def _busy_state_tail_dependent_mu(self, occ_idx_full: int, j: int, t: float, cache: dict) -> float:
        """EPIC-072: exact P(W > t) contribution (all-c-busy case), batch-size-DEPENDENT
        mu, gamma == 0 -- see `_busy_state_moments_dependent_mu`."""
        key = (occ_idx_full, j)
        cached = cache.get(key)
        if cached is not None:
            return cached
        self._ensure_full_occ_index()
        subgen, m, start, _ = _occupancy_busy_chain(
            self._occ_full_list, self._occ_full_index, self.mu_fn, occ_idx_full, j, self.a, self.b, self.l
        )
        cached = float(spla.expm_multiply(subgen * t, np.ones(m))[start])
        cache[key] = cached
        return cached

    def _abandon_chain_for_state(self, j: int, start_idle: bool = False):
        """EPIC-070: build the abandonment_chain for a given observed state (R=j ahead of
        tagged), reusing the same `c*mu` aggregate-rate simplification as
        `_busy_state_moments`/`get_tail` -- gamma>0 is natively supported by
        `abandonment_chain` (built for EPIC-068), no new construction needed here."""
        rate = self.c * self.mu_fn(1)
        if start_idle:
            return abandonment_chain(0.0, 0, rate, 0, j, self.a, self.b, self.l, self.gamma, start_idle=True)
        return abandonment_chain(rate, 1, rate, 1, j, self.a, self.b, self.l, self.gamma, start_idle=False)

    def get_abandonment_prob(self) -> float:
        """
        EPIC-070: exact probability that a PASTA-arriving (tagged) customer abandons
        (Markovian patience, rate `gamma`) before their own batch starts service. 0.0
        when `gamma == 0`. ONLY for batch-size-independent mu (see
        `_require_constant_mu`) -- see module docstring for why this combination needed
        no new construction beyond threading `gamma` through the already-`gamma`-aware
        `abandonment_chain`.
        """
        if self.gamma <= 0:
            return 0.0
        self._require_constant_mu("get_abandonment_prob")
        pi = self._solve_pi()
        a = self.a
        cache: dict = {}
        p_abandon = 0.0
        for oi, j in self._states:
            p = pi[self._state_index[(oi, j)]]
            if p <= 0:
                continue
            idle = self.c - sum(self._occ_list[oi])
            if idle >= 1 and j == a - 1:
                continue  # W=0 deterministically, can't abandon
            key = ("idle", j) if idle >= 1 else ("busy", j)
            cached = cache.get(key)
            if cached is None:
                subgen, m, start, _ = self._abandon_chain_for_state(j, start_idle=idle >= 1)
                alpha = np.zeros(m)
                alpha[start] = 1.0
                x = spla.spsolve((-subgen).tocsc(), np.ones(m))
                cached = self.gamma * float(alpha @ x)
                cache[key] = cached
            p_abandon += p * cached
        return p_abandon

    def _get_w_with_abandonment(self, num: int = 4) -> list[float]:
        """EPIC-070: exact raw moments of W conditional on being served, gamma > 0."""
        pi = self._solve_pi()
        a = self.a
        cache: dict = {}
        numerator = np.zeros(num)
        p_served_total = 0.0
        for oi, j in self._states:
            p = pi[self._state_index[(oi, j)]]
            if p <= 0:
                continue
            idle = self.c - sum(self._occ_list[oi])
            if idle >= 1 and j == a - 1:
                p_served_total += p
                continue
            key = ("idle", j) if idle >= 1 else ("busy", j)
            cached = cache.get(key)
            if cached is None:
                subgen, m, start, serve_rate = self._abandon_chain_for_state(j, start_idle=idle >= 1)
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
            numerator += p * np.array(moments)
            p_served_total += p * p_served_state
        if p_served_total <= 0:
            return [0.0] * num
        return list(numerator / p_served_total)

    def get_w(self, num: int = 4) -> list[float]:
        """
        Exact raw moments of W (waiting time), any `1 <= a <= b`, `c >= 1`, any `mu`
        (constant OR batch-size-dependent, EPIC-072) -- PROVIDED `gamma == 0`; at
        `gamma > 0` this still requires batch-size-independent mu (see
        `_require_constant_mu` -- combining dependent mu with abandonment remains a
        separate, unimplemented reserve). PASTA: a tagged arrival sees the
        stationary (occupancy, j). If some server is idle (j < a by the state
        space's own invariant), the wait is a pure Poisson(lambda) refill race, IDENTICAL
        in form to `BulkServiceMM1Calc`'s own idle branch (idle-dispatch only depends on
        the queue reaching `a`, not on how many servers are idle or what any busy server is
        doing). If all `c` servers are busy, see `_busy_state_moments`
        (batch-size-independent mu) / `_busy_state_moments_dependent_mu` (dependent mu).
        At `gamma > 0` (EPIC-070) these are conditional on being served -- see
        `BulkServiceMM1Calc.get_w` for what that means and why.
        """
        if self.gamma > 0:
            self._require_constant_mu("get_w() with gamma > 0")
            return self._get_w_with_abandonment(num)
        pi = self._solve_pi()
        a, lam = self.a, self.l

        def exp_moments(rate: float) -> list[float]:
            return [math.factorial(k) / rate**k for k in range(1, num + 1)]

        def erlang_moments(n_terms: int, rate: float) -> list[float]:
            if n_terms <= 0:
                return [0.0] * num
            acc = exp_moments(rate)
            for _ in range(n_terms - 1):
                acc = list(conv_moments(acc, exp_moments(rate), num))
            return acc

        cache: dict = {}
        w_moments = np.zeros(num)
        for oi, j in self._states:
            p = pi[self._state_index[(oi, j)]]
            if p <= 0:
                continue
            occ = self._occ_list[oi]
            idle = self.c - sum(occ)
            if idle >= 1:
                wm = erlang_moments(a - j - 1, lam)
            elif self._is_constant_mu:
                wm = self._busy_state_moments(j, num, cache)
            else:
                self._ensure_full_occ_index()
                wm = self._busy_state_moments_dependent_mu(self._occ_full_index[occ], j, num, cache)
            w_moments += p * np.array(wm)
        return list(w_moments)

    def _get_tail_with_abandonment(self, t: float) -> float:
        """EPIC-070: exact P(W > t AND served) / P(served), gamma > 0."""
        pi = self._solve_pi()
        a = self.a
        cache: dict = {}
        numerator = 0.0
        p_served_total = 0.0
        for oi, j in self._states:
            p = pi[self._state_index[(oi, j)]]
            if p <= 0:
                continue
            idle = self.c - sum(self._occ_list[oi])
            if idle >= 1 and j == a - 1:
                p_served_total += p
                continue
            key = ("idle", j) if idle >= 1 else ("busy", j)
            cached = cache.get(key)
            if cached is None:
                subgen, m, start, serve_rate = self._abandon_chain_for_state(j, start_idle=idle >= 1)
                alpha = np.zeros(m)
                alpha[start] = 1.0
                neg_a = (-subgen).tocsc()
                v = spla.spsolve(neg_a, serve_rate)
                p_served_state = 1.0 - self.gamma * float(alpha @ spla.spsolve(neg_a, np.ones(m)))
                cached = (subgen, v, start, p_served_state)
                cache[key] = cached
            subgen, v, start, p_served_state = cached
            numerator += p * float(spla.expm_multiply(subgen * t, v)[start])
            p_served_total += p * p_served_state
        if p_served_total <= 0:
            return 0.0
        return numerator / p_served_total

    def get_tail(self, t: float) -> float:
        """Exact P(W > t), any `1 <= a <= b`, `c >= 1`, any `mu` (constant OR
        batch-size-dependent, EPIC-072) -- PROVIDED `gamma == 0`; at `gamma > 0` this
        still requires batch-size-independent mu (see `_require_constant_mu`). Same
        decomposition as `get_w`, via sparse matrix-exponential-action instead of
        moment convolution. At `gamma > 0` (EPIC-070) this is P(W > t | served) --
        see `BulkServiceMM1Calc.get_tail`."""
        if t < 0:
            raise ValueError("t must be nonnegative")
        if self.gamma > 0:
            self._require_constant_mu("get_tail() with gamma > 0")
            return self._get_tail_with_abandonment(t)
        pi = self._solve_pi()
        a, lam = self.a, self.l
        cache: dict = {}

        tail = 0.0
        for oi, j in self._states:
            p = pi[self._state_index[(oi, j)]]
            if p <= 0:
                continue
            occ = self._occ_list[oi]
            idle = self.c - sum(occ)
            if idle >= 1:
                need = a - j - 1
                if need > 0:
                    key = ("idle", need)
                    cached = cache.get(key)
                    if cached is None:
                        rates = np.full(need, lam)
                        subgen = sp.diags([-rates, rates[:-1]], [0, 1], format="csc")
                        cached = float(spla.expm_multiply(subgen * t, np.ones(need))[0])
                        cache[key] = cached
                    tail += p * cached
            elif self._is_constant_mu:
                cached = cache.get(j)
                if cached is None:
                    rate = self.c * self.mu_fn(1)
                    subgen, m, start, _ = abandonment_chain(rate, 1, rate, 1, j, a, self.b, lam, 0.0, start_idle=False)
                    cached = float(spla.expm_multiply(subgen * t, np.ones(m))[start])
                    cache[j] = cached
                tail += p * cached
            else:
                self._ensure_full_occ_index()
                cached = self._busy_state_tail_dependent_mu(self._occ_full_index[occ], j, t, cache)
                tail += p * cached
        return tail

    def get_cdf(self, t: float) -> float:
        """Exact P(W <= t) -- see `get_tail` for scope."""
        return 1.0 - self.get_tail(t)

    def run(self) -> QueueResults:
        """Solve the CTMC; return E[N]-based sojourn time and average server utilization."""
        start = self._measure_time()
        pi = self._solve_pi()

        n_moments = self.get_n_moments(num=1)
        e_n = n_moments[0]
        e_t = e_n / self.l

        busy_servers = np.array([sum(self._occ_list[oi]) for oi, _ in self._states])
        e_busy = float((pi * busy_servers).sum())

        can_compute_w = self._is_constant_mu or self.gamma == 0  # EPIC-072: dependent mu needs gamma == 0
        w = self.get_w(num=1) if can_compute_w else [0, 0, 0, 0]
        res = QueueResults(
            v=[e_t, 0, 0, 0],
            w=list(w) + [0] * (4 - len(w)),
            utilization=e_busy / self.c,
        )
        self._set_duration(res, start)
        return res
