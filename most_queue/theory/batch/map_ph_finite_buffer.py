"""
Exact waiting-time and sojourn-time distribution for the finite-buffer
bulk-service queue with correlated (MAP) arrivals: MAP/PH^(a,b)/1/N.

The model is that of
    Banik A.D., Chaudhry M.L., Barik S., Singh G., "On the Heuristic
    Computational Procedures of the Virtual Waiting-Time Distribution in a
    Non-renewal Input Finite-Buffer Bulk-Service Queues: MAP/R^(a,b)/1/N",
    Journal of the Indian Society for Probability and Statistics 26 (2025)
    585-630, doi:10.1007/s41096-025-00232-0,

and, for the Poisson special case, of its open-access predecessor
    Chaudhry M.L., Banik A.D., Barik S., Goswami V., "A Novel Computational
    Procedure for the Waiting-Time Distribution (In the Queue) for
    Bulk-Service Finite-Buffer Queues with Poisson Input", Mathematics 11(5)
    (2023) 1142, doi:10.3390/math11051142 (CC BY 4.0).

**The model is theirs.** What differs here is the route to the waiting time,
and that difference is the point of this module.

Why a different route. Both papers reach the waiting time through transforms:
the queue-length distribution gives a probability generating function, a
functional relation turns it into the Laplace-Stieltjes transform of the
queueing time, and the transform is then inverted numerically. For the Poisson
case that LST is exact and elementary -- equation (39) of the 2023 paper is a
finite sum of powers of ``(1 - s/lambda')`` -- but it still has to be inverted,
and the authors do that with a Pade rational approximation (computed with
Maple's ``pade``), then a partial-fraction expansion. The 2025 MAP paper calls
its procedures heuristic in its own title for the same reason: the
approximation lives entirely in the inversion step, not in the model.

This module never forms a transform, so there is nothing to invert. The
waiting time is built directly as a phase-type distribution by following a
TAGGED customer through an absorbing Markov chain, which gives the CDF as
``1 - alpha exp(T t) 1`` and the moments as ``k! alpha (-T)^-k 1`` -- exact to
machine precision, with no Pade step, no root finding and no inversion
tolerance to tune.

The model. Customers arrive as a MAP ``(D0, D1)``. The queue holds at most
``N`` waiting customers, not counting the batch in service; an arrival finding
``N`` waiting is lost. The server works under the general ``(a, b)`` bulk rule:
it stays idle until ``a`` customers have accumulated, then takes
``min(queue, b)`` of them as one batch, and the batch's service time is
PH-distributed ``(beta, S)`` regardless of how many customers it contains.
Everyone in a batch leaves when the batch finishes, so the sojourn time is the
queueing time plus one whole batch service.

Why the tagged chain stays small. Follow a customer that joins at position
``p`` in the queue. At every service completion the server removes exactly
``b`` from the front whenever more than ``b`` are waiting, so ``p`` falls by
``b`` and nothing else about the queue matters -- if ``p > b`` then the queue
is at least ``p > b >= a``, so the server certainly starts another batch. The
queue length only ever matters at the single completion where ``p <= b``, to
decide whether the server can start at once or must idle until ``a`` have
accumulated. That check needs the number of customers BEHIND the tagged one
only up to ``a - 1``: any more and the queue is certainly at least ``a``. So
the chain is indexed by position (at most ``N``) and by a count capped at
``a``, rather than by the full queue length, which keeps it ``O(N*a*m_s*m_a)``
instead of ``O(N^2*m_s*m_a)``.

Arrivals see the system through the MAP's arrival-epoch distribution,
``pi D1 / (pi D1 1)``, which reduces to PASTA when the MAP is a Poisson
process.
"""

import time
from dataclasses import dataclass, field
from math import comb

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply, spsolve

from most_queue.random.map_ph import MAPParams, PHParams
from most_queue.structs import QueueResults

_TOL = 1e-12


@dataclass
class BulkMapPhResults(QueueResults):
    """Outcome of the exact analysis, on top of the common queue fields."""

    loss_probability: float = 0.0  # an arrival finds the buffer full
    queue_mean: float = 0.0  # mean number waiting (not counting the batch in service)
    server_busy_probability: float = 0.0
    effective_rate: float = 0.0  # admitted arrivals per unit time
    waiting_atom: float = 0.0  # P(wait = 0): the batch starts on arrival
    states: int = 0  # size of the system chain
    tagged_states: int = 0  # size of the tagged-customer chain
    p_idle: list[float] = field(default_factory=list)  # server idle, n waiting (n < a)
    p_busy: list[float] = field(default_factory=list)  # server busy, n waiting (n <= N)


class BulkServiceMapPhCalc:
    """
    Exact MAP/PH^(a,b)/1/N bulk-service queue.

    :param a: the batch is not started until ``a`` customers have accumulated.
    :param b: at most ``b`` customers are taken into one batch (``b >= a``).
    :param capacity: buffer size ``N``, counting waiting customers only -- the
        batch in service does not occupy it. Must be at least ``a``.

    Usage::

        calc = BulkServiceMapPhCalc(a=3, b=6, capacity=200)
        calc.set_sources(map_params)      # or set_poisson_sources(6.7)
        calc.set_servers(ph_params)       # or set_exponential_servers(1.7)

        calc.get_w(num=2)       # exact raw moments of the queueing time
        calc.get_w_cdf(2.0)     # exact CDF, no transform inversion
        calc.get_v_cdf(2.0)     # sojourn = queueing + one batch service
        calc.run()              # everything at once
    """

    def __init__(self, a: int = 1, b: int = 1, capacity: int = 100):
        if a < 1:
            raise ValueError(f"a must be at least 1, got {a}")
        if b < a:
            raise ValueError(f"b must be at least a, got a={a}, b={b}")
        if capacity < a:
            raise ValueError(
                f"capacity (buffer N) must be at least a, got N={capacity}, a={a}; "
                "otherwise the server could never accumulate a full batch"
            )
        self.a = int(a)
        self.b = int(b)
        self.capacity = int(capacity)
        self.map_params: MAPParams | None = None
        self.ph_params: PHParams | None = None
        self._pi: np.ndarray | None = None
        self._tagged_cache: tuple | None = None

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def set_sources(self, map_params: MAPParams):
        """Arrival MAP ``(D0, D1)``."""
        d0 = np.asarray(map_params.D0, dtype=float)
        d1 = np.asarray(map_params.D1, dtype=float)
        if d0.shape != d1.shape or d0.ndim != 2 or d0.shape[0] != d0.shape[1]:
            raise ValueError(f"D0 and D1 must be square and of equal shape, got {d0.shape} and {d1.shape}")
        if np.any(d1 < -_TOL):
            raise ValueError("D1 must be non-negative")
        if np.max(np.abs((d0 + d1).sum(axis=1))) > 1e-8:
            raise ValueError("rows of D0 + D1 must sum to zero")
        self.map_params = MAPParams(D0=d0, D1=d1)
        self._pi = None
        self._tagged_cache = None
        return self

    def set_poisson_sources(self, rate: float):
        """Shorthand for a Poisson arrival stream, the classical ``M/...`` case."""
        if rate <= 0:
            raise ValueError(f"arrival rate must be positive, got {rate}")
        return self.set_sources(MAPParams(D0=np.array([[-rate]]), D1=np.array([[rate]])))

    def set_servers(self, ph_params: PHParams):
        """Batch service time, PH-distributed ``(beta, S)``, the same for every batch size."""
        alpha = np.asarray(ph_params.alpha, dtype=float).ravel()
        mat = np.asarray(ph_params.T, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] != alpha.size:
            raise ValueError(f"alpha of size {alpha.size} does not match T of shape {mat.shape}")
        if abs(alpha.sum() - 1.0) > 1e-8 or np.any(alpha < -_TOL):
            raise ValueError("alpha must be a probability vector")
        exit_rates = -mat.sum(axis=1)
        if np.any(exit_rates < -1e-8):
            raise ValueError("T must be a sub-generator (row sums must not be positive)")
        if np.all(exit_rates < _TOL):
            raise ValueError("the service distribution never completes: T has no exit rate")
        self.ph_params = PHParams(alpha=alpha, T=mat)
        self._pi = None
        self._tagged_cache = None
        return self

    def set_exponential_servers(self, rate: float):
        """Shorthand for exponential batch service, the classical ``.../M^(a,b)/...`` case."""
        if rate <= 0:
            raise ValueError(f"service rate must be positive, got {rate}")
        return self.set_servers(PHParams(alpha=np.array([1.0]), T=np.array([[-rate]])))

    def _check_ready(self):
        if self.map_params is None:
            raise ValueError("set_sources (or set_poisson_sources) must be called first")
        if self.ph_params is None:
            raise ValueError("set_servers (or set_exponential_servers) must be called first")

    # ------------------------------------------------------------------
    # the system chain
    # ------------------------------------------------------------------

    @property
    def _dims(self):
        d0 = np.asarray(self.map_params.D0)
        return d0.shape[0], np.asarray(self.ph_params.T).shape[0]

    def _idle_index(self, n: int, m: int) -> int:
        """Server idle with ``n`` waiting (n < a), MAP phase ``m``."""
        m_a, _ = self._dims
        return n * m_a + m

    def _busy_index(self, n: int, j: int, m: int) -> int:
        """Server busy, ``n`` waiting (n <= N), service phase ``j``, MAP phase ``m``."""
        m_a, m_s = self._dims
        return self.a * m_a + (n * m_s + j) * m_a + m

    def _system_size(self) -> int:
        m_a, m_s = self._dims
        return self.a * m_a + (self.capacity + 1) * m_s * m_a

    def _build_generator(self) -> np.ndarray:  # pylint: disable=too-many-nested-blocks
        """
        Generator of the full system chain.

        States are "idle with n waiting" for ``n < a`` and "busy with n waiting,
        in service phase j" for ``n <= N``, each carried with the MAP phase.
        """
        d0 = np.asarray(self.map_params.D0)
        d1 = np.asarray(self.map_params.D1)
        beta = np.asarray(self.ph_params.alpha)
        s_mat = np.asarray(self.ph_params.T)
        s_exit = -s_mat.sum(axis=1)
        m_a, m_s = self._dims
        a, b, cap = self.a, self.b, self.capacity

        size = self._system_size()
        gen = np.zeros((size, size))

        for n in range(a):
            for m in range(m_a):
                src = self._idle_index(n, m)
                for m2 in range(m_a):  # phase changes without an arrival
                    if m2 != m:
                        gen[src, self._idle_index(n, m2)] += d0[m, m2]
                for m2 in range(m_a):  # an arrival
                    rate = d1[m, m2]
                    if rate <= 0.0:
                        continue
                    if n + 1 < a:
                        gen[src, self._idle_index(n + 1, m2)] += rate
                    else:  # the a-th customer starts the batch, leaving an empty queue
                        for j2 in range(m_s):
                            gen[src, self._busy_index(0, j2, m2)] += rate * beta[j2]

        for n in range(cap + 1):
            for j in range(m_s):
                for m in range(m_a):
                    src = self._busy_index(n, j, m)
                    for m2 in range(m_a):
                        if m2 != m:
                            gen[src, self._busy_index(n, j, m2)] += d0[m, m2]
                    for m2 in range(m_a):  # arrival; lost when the buffer is full
                        rate = d1[m, m2]
                        if rate > 0.0:
                            gen[src, self._busy_index(min(n + 1, cap), j, m2)] += rate
                    for j2 in range(m_s):  # service phase change
                        if j2 != j:
                            gen[src, self._busy_index(n, j2, m)] += s_mat[j, j2]
                    done = s_exit[j]  # the batch finishes
                    if done <= 0.0:
                        continue
                    if n >= a:
                        left = n - min(n, b)
                        for j2 in range(m_s):
                            gen[src, self._busy_index(left, j2, m)] += done * beta[j2]
                    else:
                        gen[src, self._idle_index(n, m)] += done

        np.fill_diagonal(gen, 0.0)
        np.fill_diagonal(gen, -gen.sum(axis=1))
        return gen

    def _stationary(self) -> np.ndarray:
        """Stationary distribution of the system chain. Finite, so always exists."""
        if self._pi is not None:
            return self._pi
        self._check_ready()
        gen = self._build_generator()
        size = gen.shape[0]
        mat = gen.T.copy()
        mat[-1, :] = 1.0  # replace one balance equation by the normalisation
        rhs = np.zeros(size)
        rhs[-1] = 1.0
        try:
            pi = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:  # pragma: no cover - only for degenerate input
            pi = np.linalg.lstsq(np.vstack([gen.T[:-1], np.ones(size)]), rhs, rcond=None)[0]
        pi = np.maximum(pi, 0.0)
        pi /= pi.sum()
        self._pi = pi
        return pi

    def get_p(self) -> dict:
        """
        Time-stationary state probabilities, aggregated over phases.

        :return: ``idle[n]`` for ``n < a`` (server idle with ``n`` waiting) and
            ``busy[n]`` for ``n <= N`` (server busy with ``n`` waiting). These
            are the paper's ``p0(n)`` and ``p1(n)``.
        """
        pi = self._stationary()
        m_a, m_s = self._dims
        idle = [float(sum(pi[self._idle_index(n, m)] for m in range(m_a))) for n in range(self.a)]
        busy = [
            float(sum(pi[self._busy_index(n, j, m)] for j in range(m_s) for m in range(m_a)))
            for n in range(self.capacity + 1)
        ]
        return {"idle": idle, "busy": busy}

    def _arrival_epoch_weights(self) -> np.ndarray:
        """
        Distribution seen by an arriving customer: ``pi D1``, normalised.

        For a Poisson stream this is ``pi`` itself (PASTA); for a general MAP
        it is not, and using ``pi`` instead would be a silent modelling error.
        The vector is indexed by the state LEFT BEHIND by the transition, i.e.
        the post-arrival MAP phase with the pre-arrival queue state.
        """
        pi = self._stationary()
        d1 = np.asarray(self.map_params.D1)
        m_a, m_s = self._dims
        weights = np.zeros((self._system_size(), m_a))
        for n in range(self.a):
            for m in range(m_a):
                src = self._idle_index(n, m)
                weights[src] += pi[src] * d1[m]
        for n in range(self.capacity + 1):
            for j in range(m_s):
                for m in range(m_a):
                    src = self._busy_index(n, j, m)
                    weights[src] += pi[src] * d1[m]
        total = weights.sum()
        if total <= 0.0:
            raise ValueError("the arrival process never fires: D1 is identically zero")
        return weights / total

    def get_loss_probability(self) -> float:
        """Probability that an arriving customer finds the buffer full."""
        weights = self._arrival_epoch_weights()
        m_a, m_s = self._dims
        return float(sum(weights[self._busy_index(self.capacity, j, m)].sum() for j in range(m_s) for m in range(m_a)))

    # ------------------------------------------------------------------
    # the tagged-customer chain
    # ------------------------------------------------------------------

    def _tagged_layout(self):
        """
        Index the absorbing chain followed by one tagged customer.

        Two blocks. ``BUSY(p, d, j, m)``: the server is busy in phase ``j``,
        the tagged customer is ``p``-th in the queue and ``d`` customers sit
        behind it, counted only up to ``a - 1`` because beyond that the queue
        is certainly long enough to start a batch. ``IDLE(k, m)``: the server
        is idle and ``k`` more arrivals are needed before the batch -- which
        will contain the tagged customer -- can start.
        """
        m_a, m_s = self._dims
        cap, a = self.capacity, self.a
        busy_count = cap * a * m_s * m_a  # p = 1..N, d = 0..a-1
        idle_count = max(a - 1, 0) * m_a  # k = 1..a-1

        def busy(p, d, j, m):
            return ((((p - 1) * a + d) * m_s) + j) * m_a + m

        def idle(k, m):
            return busy_count + (k - 1) * m_a + m

        return busy, idle, busy_count, busy_count + idle_count

    def _build_tagged(self) -> np.ndarray:
        """
        Sub-generator of the tagged-customer chain.

        Absorption happens exactly when the tagged customer enters service, so
        the time to absorption IS its queueing time.

        Off-diagonal entries are written first and each state's absorption rate
        is accumulated alongside them; only then is the diagonal set to minus
        their total. Doing it that way rather than from the state's total rate
        matters: once ``d`` has reached its cap a further arrival leads back to
        the very same state, and that self-loop has to be dropped rather than
        counted as leaving -- counted, it would masquerade as absorption and
        silently shorten every waiting time.
        """
        self._check_ready()
        d0 = np.asarray(self.map_params.D0)
        d1 = np.asarray(self.map_params.D1)
        beta = np.asarray(self.ph_params.alpha)
        s_mat = np.asarray(self.ph_params.T)
        s_exit = -s_mat.sum(axis=1)
        m_a, m_s = self._dims
        a, b, cap = self.a, self.b, self.capacity

        busy, idle, _, size = self._tagged_layout()
        sub = np.zeros((size, size))
        absorb = np.zeros(size)

        def add(src, dst, rate):
            if rate > 0.0 and dst != src:  # a self-loop is not a transition
                sub[src, dst] += rate

        # The four loops are the four coordinates of a state -- position,
        # customers behind, service phase, arrival phase -- so the nesting is
        # the state index rather than incidental structure.
        for p in range(1, cap + 1):  # pylint: disable=too-many-nested-blocks
            for d in range(a):
                for j in range(m_s):
                    for m in range(m_a):
                        src = busy(p, d, j, m)
                        for m2 in range(m_a):
                            if m2 != m:
                                add(src, busy(p, d, j, m2), d0[m, m2])
                        # The buffer holds p + d customers while d is exact,
                        # so it is full exactly when that reaches N. A blocked
                        # arrival still advances the MAP phase -- it just does
                        # not join the queue.
                        blocked = (d < a - 1) and (p + d >= cap)
                        for m2 in range(m_a):
                            # Someone joins behind the tagged customer. Once the
                            # count reaches a - 1 the queue is certainly long
                            # enough to start a batch, so tracking it further
                            # would change nothing the tagged customer can feel.
                            nxt = d if blocked else min(d + 1, a - 1)
                            add(src, busy(p, nxt, j, m2), d1[m, m2])
                        for j2 in range(m_s):
                            if j2 != j:
                                add(src, busy(p, d, j2, m), s_mat[j, j2])
                        done = s_exit[j]
                        if done <= 0.0:
                            continue
                        if p > b:
                            # The server takes b from the front and starts
                            # again: the queue holds at least p > b >= a, so it
                            # cannot fall idle here.
                            for j2 in range(m_s):
                                add(src, busy(p - b, d, j2, m), done * beta[j2])
                        elif p + d >= a:
                            absorb[src] += done  # the tagged customer is in this batch
                        else:
                            add(src, idle(a - p - d, m), done)

        for k in range(1, a):
            for m in range(m_a):
                src = idle(k, m)
                for m2 in range(m_a):
                    if m2 != m:
                        add(src, idle(k, m2), d0[m, m2])
                for m2 in range(m_a):
                    if k > 1:
                        add(src, idle(k - 1, m2), d1[m, m2])
                    else:
                        absorb[src] += d1[m, m2]  # this arrival starts the batch

        np.fill_diagonal(sub, 0.0)
        np.fill_diagonal(sub, -(sub.sum(axis=1) + absorb))
        return sub

    def _initial_vector(self):
        """
        Where a tagged customer starts, and how often it starts nowhere.

        Blocked arrivals are dropped and the rest renormalised, so the waiting
        time is conditional on being admitted -- the only customers that have
        one.
        """
        weights = self._arrival_epoch_weights()
        m_a, m_s = self._dims
        a, cap = self.a, self.capacity
        busy, idle, _, size = self._tagged_layout()

        vec = np.zeros(size)
        atom = 0.0
        admitted = 0.0

        for n in range(a):
            for m in range(m_a):
                row = weights[self._idle_index(n, m)]
                if row.sum() <= 0.0:
                    continue
                admitted += row.sum()
                if n + 1 == a:
                    atom += row.sum()  # this arrival completes the batch
                else:
                    for m2 in range(m_a):
                        if row[m2] > 0.0:
                            vec[idle(a - n - 1, m2)] += row[m2]

        for n in range(cap + 1):
            if n == cap:
                continue  # blocked: no waiting time to speak of
            for j in range(m_s):
                for m in range(m_a):
                    row = weights[self._busy_index(n, j, m)]
                    if row.sum() <= 0.0:
                        continue
                    admitted += row.sum()
                    for m2 in range(m_a):
                        if row[m2] > 0.0:
                            vec[busy(n + 1, 0, j, m2)] += row[m2]

        if admitted <= 0.0:
            raise ValueError("every arrival is blocked; the waiting time is undefined")
        return vec / admitted, atom / admitted

    # ------------------------------------------------------------------
    # waiting and sojourn time
    # ------------------------------------------------------------------

    def _tagged(self):
        """
        The tagged chain, its starting vector and its atom at zero, built once.

        Rebuilding per time point is what makes a CDF sweep slow, so the result
        is cached and invalidated whenever the arrival or service law changes.
        """
        if self._tagged_cache is None:
            sub = self._build_tagged()
            vec, atom = self._initial_vector()
            self._tagged_cache = (sub, vec, atom)
        return self._tagged_cache

    def get_w(self, num: int = 4) -> list[float]:
        """
        Raw moments of the queueing time of an admitted customer.

        Exact: solves ``(-T) m_k = k m_{k-1}`` on the tagged chain, which is
        the phase-type moment identity ``E[W^k] = k! alpha (-T)^-k 1``. No
        transform is formed and none is inverted.
        """
        if num < 1:
            raise ValueError(f"num must be at least 1, got {num}")
        sub, vec, _ = self._tagged()
        neg = csc_matrix(-sub)
        ones = np.ones(sub.shape[0])
        moments, current = [], ones
        for k in range(1, num + 1):
            current = spsolve(neg, k * current)
            moments.append(float(vec @ current))
        return moments

    def get_v(self, num: int = 4) -> list[float]:
        """
        Raw moments of the sojourn time.

        Everyone in a batch leaves when the batch finishes, so the sojourn time
        is the queueing time plus one whole batch service, and the two are
        independent -- the batch a customer joins has not started when its wait
        ends. The moments therefore combine binomially.
        """
        waiting = self.get_w(num)
        service = self.get_service_moments(num)
        out = []
        for k in range(1, num + 1):
            total = service[k - 1] + waiting[k - 1]
            for i in range(1, k):
                total += comb(k, i) * waiting[i - 1] * service[k - i - 1]
            out.append(float(total))
        return out

    def get_service_moments(self, num: int = 4) -> list[float]:
        """Raw moments of one batch's service time."""
        self._check_ready()
        beta = np.asarray(self.ph_params.alpha)
        mat = np.asarray(self.ph_params.T)
        neg = -mat
        ones = np.ones(mat.shape[0])
        out, current = [], ones
        for k in range(1, num + 1):
            current = np.linalg.solve(neg, k * current)
            out.append(float(beta @ current))
        return out

    def get_w_cdf(self, t):
        """
        Exact CDF of the queueing time, at one time or a whole array of them.

        ``1 - alpha exp(T t) 1``. This is the quantity the source papers reach
        by Pade-approximating a transform and inverting it; here it is a matrix
        exponential of a finite generator, with no approximation anywhere.

        The action of the exponential on a vector is computed directly rather
        than by forming ``expm(T t)``, which would be both far slower and
        needlessly dense for the chains this model produces.
        """
        times = np.atleast_1d(np.asarray(t, dtype=float))
        if np.any(times < 0):
            raise ValueError(f"t must be non-negative, got {t}")
        sub, vec, atom = self._tagged()
        ones = np.ones(sub.shape[0])
        out = np.empty(times.size)
        for i, moment in enumerate(times):
            if moment == 0.0:
                out[i] = atom
                continue
            survival = float(vec @ expm_multiply(csc_matrix(sub * moment), ones))
            out[i] = min(max(1.0 - survival, 0.0), 1.0)
        return float(out[0]) if np.isscalar(t) or np.ndim(t) == 0 else out

    def get_w_tail(self, t):
        """``P(queueing time > t)``."""
        return 1.0 - np.asarray(self.get_w_cdf(t))

    def get_v_cdf(self, t):
        """
        Exact CDF of the sojourn time, at one time or a whole array of them.

        The sojourn time is the queueing time followed by one whole batch
        service, so the two phase-type chains are stacked: the waiting chain
        absorbs into the service chain rather than into the exit.
        """
        times = np.atleast_1d(np.asarray(t, dtype=float))
        if np.any(times < 0):
            raise ValueError(f"t must be non-negative, got {t}")
        sub, vec, atom = self._tagged()
        beta = np.asarray(self.ph_params.alpha)
        s_mat = np.asarray(self.ph_params.T)
        n_w, n_s = sub.shape[0], s_mat.shape[0]

        exit_rates = -sub.sum(axis=1)
        joint = np.zeros((n_w + n_s, n_w + n_s))
        joint[:n_w, :n_w] = sub
        joint[:n_w, n_w:] = np.outer(exit_rates, beta)
        joint[n_w:, n_w:] = s_mat

        start = np.zeros(n_w + n_s)
        start[:n_w] = vec
        start[n_w:] = atom * beta  # waited no time at all, straight into service

        ones = np.ones(n_w + n_s)
        out = np.empty(times.size)
        for i, moment in enumerate(times):
            if moment == 0.0:
                out[i] = 0.0
                continue
            survival = float(start @ expm_multiply(csc_matrix(joint * moment), ones))
            out[i] = min(max(1.0 - survival, 0.0), 1.0)
        return float(out[0]) if np.isscalar(t) or np.ndim(t) == 0 else out

    def get_v_tail(self, t):
        """``P(sojourn time > t)``."""
        return 1.0 - np.asarray(self.get_v_cdf(t))

    def run(self, num: int = 2) -> BulkMapPhResults:
        """Everything in one pass."""
        start = time.process_time()
        probs = self.get_p()
        loss = self.get_loss_probability()
        waiting = self.get_w(num)
        sojourn = self.get_v(num)
        _, _, atom = self._tagged()

        d1 = np.asarray(self.map_params.D1)
        pi_phase = self._map_stationary()
        arrival_rate = float(pi_phase @ d1 @ np.ones(d1.shape[0]))
        queue_mean = sum(n * p for n, p in enumerate(probs["idle"])) + sum(n * p for n, p in enumerate(probs["busy"]))
        busy_prob = float(sum(probs["busy"]))

        _, _, _, tagged_size = self._tagged_layout()
        return BulkMapPhResults(
            v=sojourn,
            w=waiting,
            p=probs["idle"] + probs["busy"],
            utilization=busy_prob,
            loss_probability=loss,
            queue_mean=queue_mean,
            server_busy_probability=busy_prob,
            effective_rate=arrival_rate * (1.0 - loss),
            waiting_atom=atom,
            states=self._system_size(),
            tagged_states=tagged_size,
            p_idle=probs["idle"],
            p_busy=probs["busy"],
            duration=time.process_time() - start,
        )

    def _map_stationary(self) -> np.ndarray:
        """Stationary phase distribution of the MAP itself."""
        gen = np.asarray(self.map_params.D0) + np.asarray(self.map_params.D1)
        size = gen.shape[0]
        mat = np.vstack([gen.T[:-1], np.ones(size)])
        rhs = np.zeros(size)
        rhs[-1] = 1.0
        vec = np.linalg.lstsq(mat, rhs, rcond=None)[0]
        return vec / vec.sum()

    def get_arrival_rate(self) -> float:
        """Nominal arrival rate of the MAP, before any blocking."""
        self._check_ready()
        d1 = np.asarray(self.map_params.D1)
        return float(self._map_stationary() @ d1 @ np.ones(d1.shape[0]))

    def __repr__(self) -> str:
        return f"BulkServiceMapPhCalc(a={self.a}, b={self.b}, capacity={self.capacity})"
