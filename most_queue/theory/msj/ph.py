"""Small FCFS multiserver-job systems with general phase-type service.

The sequence-state CTMC is exact for an N-job loss system; increasing N
approximates the open queue. Stability is checked using the saturated CTMC.
This is NOT the matrix-geometric algorithm of Anggraito et al. (2025),
doi:10.1016/j.peva.2025.102486. For PH saturated systems see Grosof et al.,
The RESET and MARC techniques, with application to multiserver-job analysis,
Performance Evaluation (2023), doi:10.1016/j.peva.2023.102378.
"""

import warnings
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from most_queue.random.map_ph import PHParams
from most_queue.structs import MsjResults
from most_queue.theory.base_queue import BaseQueue


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checked_ph(params):
    """Copy and validate a proper real PH, including transience of every phase."""
    if np.iscomplexobj(params.alpha) or np.iscomplexobj(params.T):
        raise ValueError("PH parameters must be real, not complex moment fits")
    alpha, matrix = np.array(params.alpha, dtype=float), np.array(params.T, dtype=float)
    if alpha.ndim != 1 or not alpha.size or matrix.shape != (alpha.size, alpha.size):
        raise ValueError("PH alpha and T must have compatible nonempty dimensions")
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(matrix)):
        raise ValueError("PH parameters must be finite")
    if np.any(alpha < 0) or not np.isclose(alpha.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("PH alpha must be a probability vector without mass at zero")
    offdiag = matrix - np.diag(np.diag(matrix))
    if np.any(np.diag(matrix) >= 0) or np.any(offdiag < 0):
        raise ValueError("PH T needs negative diagonal and nonnegative off-diagonal entries")
    scale = float(np.max(-np.diag(matrix)))
    if np.any(matrix.sum(axis=1) > 1e-12 * scale):
        raise ValueError("PH T row sums must be nonpositive")
    if np.max(np.linalg.eigvals(matrix / scale).real) >= 0:
        raise ValueError("PH T must be transient (all phases eventually absorb)")
    mean = float(alpha @ np.linalg.solve(-matrix, np.ones(alpha.size)))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("PH mean must be finite and positive")
    return PHParams(alpha=alpha / alpha.sum(), T=matrix), mean


class MsjPHCalc(BaseQueue):
    """Enumerate a bounded FCFS PH-MSJ CTMC and its saturated counterpart.

    Configure positive class arrival rates with ``set_sources`` and integer
    server needs plus PHParams with ``set_servers``. ``truncation`` limits the
    TOTAL number of jobs, not just waiting jobs. ``max_states`` applies to both
    chains. Results use the admitted rates in Little's law. Higher moments and
    analytical tail probabilities are not implemented.
    """

    def __init__(self, k: int, truncation: int = 12, max_states: int = 50_000):
        super().__init__(n=_positive_int(k, "k"))
        self.N = _positive_int(truncation, "truncation")
        self.max_states = _positive_int(max_states, "max_states")
        self.rates = None
        self.needs = None
        self.services = None
        self.means = None
        self.result = None

    def set_sources(self, rates):
        """Set strictly positive, finite Poisson rates (one per class)."""
        rates = np.asarray(rates, dtype=float)
        if rates.ndim != 1 or not rates.size or not np.all(np.isfinite(rates)) or np.any(rates <= 0):
            raise ValueError("rates must be a nonempty vector of finite positive rates")
        self.rates = rates.copy()
        self.is_sources_set = True
        self._invalidate()

    def set_servers(self, needs, services: list[PHParams]):
        """Set simultaneous resource needs and a proper real PH for each class."""
        needs = [_positive_int(need, "server need") for need in needs]
        if not needs or max(needs) > self.n or len(needs) != len(services):
            raise ValueError("one service per class is required, with 1 <= need <= k")
        checked = [_checked_ph(params) for params in services]
        self.needs = needs
        self.services = [item[0] for item in checked]
        self.means = np.array([item[1] for item in checked])
        self.is_servers_set = True
        self._invalidate()

    def _invalidate(self):
        self.result = None
        self.p = self.w = self.v = self.ro = None

    def _validate(self):
        self._check_if_servers_and_sources_set()
        if len(self.rates) != len(self.needs):
            raise ValueError("arrival and service class counts must match")

    def _fill(self, seq):
        """Branch on initial phases of the maximal fitting FCFS prefix."""
        free = self.n - sum(self.needs[c] for c, phase in seq if phase >= 0)
        for pos, (cls, phase) in enumerate(seq):
            if phase >= 0:
                continue
            if self.needs[cls] > free:
                break
            for initial, prob in enumerate(self.services[cls].alpha):
                if prob > 0:
                    started = seq[:pos] + ((cls, initial),) + seq[pos + 1 :]
                    for state, weight in self._fill(started):
                        yield state, prob * weight
            return
        yield seq, 1.0

    def _service_moves(self, seq):
        for pos, (cls, phase) in enumerate(seq):
            if phase < 0:
                continue
            row = self.services[cls].T[phase]
            for nxt, rate in enumerate(row):
                if nxt != phase and rate > 0:
                    yield seq[:pos] + ((cls, nxt),) + seq[pos + 1 :], float(rate), False
            absorption = float(-row.sum())
            if absorption > 0:
                yield seq[:pos] + seq[pos + 1 :], absorption, True

    def _open_moves(self, seq):
        if len(seq) < self.N:
            for cls, rate in enumerate(self.rates):
                for target, prob in self._fill(seq + ((cls, -1),)):
                    yield target, rate * prob
        for target, rate, completed in self._service_moves(seq):
            if completed:
                for filled, prob in self._fill(target):
                    yield filled, rate * prob
            else:
                yield target, rate

    def _draw(self, active):
        """Fill saturated resources with iid classes until a head is blocked."""
        frontier = {tuple(sorted(active)): 1.0}
        blocked = defaultdict(float)
        mix = self.rates / self.rates.sum()
        # Merge permutations at every level; enumerating draw paths would be
        # exponential even when the resulting multiset state space is small.
        while frontier:
            following = defaultdict(float)
            for busy, weight in frontier.items():
                free = self.n - sum(self.needs[c] for c, _ in busy)
                for cls, prob in enumerate(mix):
                    if self.needs[cls] > free:
                        blocked[(busy, cls)] += weight * prob
                    else:
                        for phase in np.flatnonzero(self.services[cls].alpha):
                            target = tuple(sorted(busy + ((cls, int(phase)),)))
                            following[target] += weight * prob * self.services[cls].alpha[phase]
                    if len(following) + len(blocked) > self.max_states:
                        raise ValueError("MSJ max_states exceeded during saturated fill; reduce phases, classes or k")
            frontier = following
        yield from blocked.items()

    def _saturated_moves(self, state):
        active, blocked = state
        for target, rate, completed in self._service_moves(active):
            free = self.n - sum(self.needs[c] for c, _ in target)
            if completed and self.needs[blocked] <= free:
                for phase, prob in enumerate(self.services[blocked].alpha):
                    if prob > 0:
                        for filled, weight in self._draw(target + ((blocked, phase),)):
                            yield filled, rate * prob * weight
            else:
                yield (tuple(sorted(target)), blocked), rate

    def _solve(self, initial, moves):
        states, index, edges = [], {}, []

        def register(state):
            if state not in index:
                if len(states) >= self.max_states:
                    raise ValueError("MSJ max_states exceeded; reduce truncation, phases, classes or k")
                index[state] = len(states)
                states.append(state)
            return index[state]

        for state in initial:
            register(state)
        head = 0
        while head < len(states):
            targets = defaultdict(float)
            for target, rate in moves(states[head]):
                targets[register(target)] += rate
            for target, rate in targets.items():
                if target != head:
                    edges.append((head, target, rate))
            head += 1
        size = len(states)
        rows, cols, rates = zip(*edges) if edges else ([], [], [])
        q = coo_matrix((rates, (rows, cols)), shape=(size, size)).tocsr()
        q -= diags(np.asarray(q.sum(axis=1)).ravel())
        a = q.T.tolil()
        a[0, :] = 1
        rhs = np.zeros(size)
        rhs[0] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            pi = spsolve(a.tocsc(), rhs)
        if not np.all(np.isfinite(pi)) or np.min(pi) < -1e-9:
            raise ArithmeticError("MSJ stationary solver returned invalid probabilities")
        pi = np.maximum(pi, 0)
        pi /= pi.sum()
        residual = float(np.max(np.abs(q.T @ pi)))
        if residual > 1e-9 * max(1.0, float(np.max(-q.diagonal()))):
            raise ArithmeticError("MSJ stationary residual exceeds tolerance")
        return states, pi, residual

    def saturated_throughput(self) -> float:
        """Return PH-FCFS saturated throughput for the configured class mix."""
        self._validate()
        states, pi, _ = self._solve((state for state, _ in self._draw(())), self._saturated_moves)
        completion_rates = [sum(-self.services[c].T[phase].sum() for c, phase in active) for active, _ in states]
        return float(pi @ completion_rates)

    def run(self) -> MsjResults:
        """Check open FCFS stability, then solve the N-job truncated system."""
        self._validate()
        start = self._measure_time()
        threshold = self.saturated_throughput()
        total_rate = float(self.rates.sum())
        if total_rate >= threshold:
            raise ValueError(f"FCFS MSJ is unstable: arrival rate {total_rate:g} >= saturated throughput {threshold:g}")
        states, pi, residual = self._solve([()], self._open_moves)
        sizes = np.array([len(seq) for seq in states])
        self.p = np.bincount(sizes, weights=pi, minlength=self.N + 1).tolist()
        boundary = self.p[-1]
        admitted = self.rates * (1 - boundary)
        count = np.zeros((len(states), len(self.rates)))
        waiting = np.zeros_like(count)
        busy = np.zeros(len(states))
        for i, seq in enumerate(states):
            for cls, phase in seq:
                count[i, cls] += 1
                if phase < 0:
                    waiting[i, cls] += 1
                else:
                    busy[i] += self.needs[cls]
        v_class = (pi @ count / admitted).tolist()
        w_class = (pi @ waiting / admitted).tolist()
        mix = self.rates / total_rate
        self.v, self.w = [float(mix @ v_class)], [float(mix @ w_class)]
        self.ro = float(pi @ busy / self.n)
        self.mean_jobs_in_system = float(pi @ count.sum(axis=1))
        self.mean_jobs_on_queue = float(pi @ waiting.sum(axis=1))
        self.result = MsjResults(
            v=self.v,
            w=self.w,
            p=self.p,
            v_per_class=v_class,
            w_per_class=w_class,
            utilization=self.ro,
            throughput=float(admitted.sum()),
            offered_load=float(self.rates @ (self.means * self.needs) / self.n),
            boundary_mass=boundary,
            stability_threshold=threshold,
            state_count=len(states),
            stationary_residual=residual,
        )
        self._set_duration(self.result, start)
        return self.result

    def get_w(self):
        """Return the mean waiting time (higher moments are not computed)."""
        return (self.result or self.run()).w

    def get_v(self):
        """Return the mean sojourn time (higher moments are not computed)."""
        return (self.result or self.run()).v

    def get_p(self):
        """Return the stationary number-in-system probabilities of the truncation."""
        return (self.result or self.run()).p
