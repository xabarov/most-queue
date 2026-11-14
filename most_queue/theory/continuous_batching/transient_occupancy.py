"""Transient FCFS occupancy-cap queue, with finite-state-projection bounds.

Poisson arrivals have a piecewise-constant rate. At population j, the total
completion rate is min(j,k)*mu(min(j,k)); excess customers wait in an infinite
FCFS queue. This is an exponential-completion surrogate for a shared serving
pool, not a token-level model of a GPU. The cap and service rates stay fixed.

Finite state projection follows Munsky and Khammash, J. Chem. Phys. 124,
044104 (2006), doi:10.1063/1.2145882. Exits from states 0..N go to an absorbing
accounting state. Kept probabilities are lower bounds; escaped mass bounds
the error of every [0,1]-valued observable. It is NEVER renormalized away.
Bounds concern model truncation, evaluated in floating point without verified
rounding. The derivation and arrival-time waiting law are in
docs/models/transient-occupancy.md.
"""

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaincc  # pylint: disable=no-name-in-module  # SciPy-generated ufunc.

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue


@dataclass
class TransientOccupancyResults(QueueResults):
    """Time-indexed subprobabilities and escaped mass; stationary fields unset.

    ``state_probabilities[i,j]`` is P(N(t_i)=j, no exit through N+1 by t_i).
    ``escape_probability`` is a cumulative first-exit probability, not loss,
    blocking, or the instantaneous probability of being outside the domain.
    """

    times: np.ndarray | None = None
    state_probabilities: np.ndarray | None = None
    escape_probability: np.ndarray | None = None
    arrival_rates: np.ndarray | None = None
    concurrency_cap: int = 0
    saturated_departure_rate: float = 0.0
    mass_residual: float = 0.0
    roundoff_certified: bool = False

    def _bounded_observable(self, values):
        lower = self.state_probabilities @ values
        return np.column_stack((lower, np.minimum(1.0, lower + self.escape_probability)))

    def population_tail_bounds(self, level: int):
        """Return [lower, upper] for P(N(t) >= level) at each requested time."""
        if isinstance(level, bool) or not isinstance(level, Integral) or level < 0:
            raise ValueError("level must be a nonnegative integer")
        if level == 0:
            return np.ones((len(self.times), 2))
        return self._bounded_observable(np.arange(self.state_probabilities.shape[1]) >= level)

    def waiting_tail_bounds(self, deadline: float):
        """Bound P(W_t > deadline) for a hypothetical arrival at each time t.

        W is time to a free active slot, excluding prefill and first decode.
        While this customer waits, all k slots are occupied. Its Erlang shape
        is j-k+1 and rate k*mu(k), unaffected by later FCFS arrivals. A single
        deadline is in the same time unit as the rate schedule. At zero this
        returns waiting-probability bounds. At lambda(t)=0 it still describes
        a hypothetical probe, not a population of real arriving customers.
        """
        if not np.isfinite(deadline) or deadline < 0:
            raise ValueError("deadline must be finite and nonnegative")
        values = np.zeros(self.state_probabilities.shape[1])
        shapes = np.arange(1, len(values) - self.concurrency_cap + 1)
        values[self.concurrency_cap :] = gammaincc(shapes, self.saturated_departure_rate * deadline)
        return self._bounded_observable(values)


class TransientOccupancyQueueCalc(BaseQueue):
    """Piecewise Poisson input to an occupancy-dependent pool with k slots.

    ``queue_truncation`` bounds TOTAL population for computation only; it is
    not a physical buffer. Temporary overload is allowed on a finite horizon.
    The initial law must be supported on 0..queue_truncation.
    """

    def __init__(self, k: int, queue_truncation: int = 200, max_states: int = 20_000):
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be a positive integer")
        super().__init__(n=int(k))
        if isinstance(queue_truncation, bool) or not isinstance(queue_truncation, Integral) or queue_truncation < k:
            raise ValueError("queue_truncation must be an integer >= k")
        if isinstance(max_states, bool) or not isinstance(max_states, Integral) or max_states < 3:
            raise ValueError("max_states must be an integer >= 3")
        if queue_truncation + 2 > max_states:
            raise ValueError("queue_truncation + 2 exceeds max_states")
        self.k, self.truncation = int(k), int(queue_truncation)
        self.schedule = None
        self.departure_rates = None

    def set_sources(self, schedule):
        """Set scalar lambda >= 0, or [(start_time, lambda), ...] starting at 0.

        Start times must strictly increase; the last rate continues forever.
        No interpolation or averaging is performed between change points.
        """
        values = np.asarray([[0.0, schedule]] if np.isscalar(schedule) else schedule, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
            raise ValueError("schedule must be a scalar or nonempty (start, rate) pairs")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("schedule times and rates must be finite and nonnegative")
        if values[0, 0] != 0 or np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError("schedule must start at zero with strictly increasing change times")
        self.schedule = values.copy()
        self.is_sources_set = True

    def set_servers(self, mu):
        """Set per-request rate mu(occupancy), or a positive scalar rate.

        A callable is evaluated once at each occupancy 1..k. Rates must be
        positive and finite; their product with occupancy must also be finite.
        """
        occupancy = np.arange(1, self.k + 1)
        values = np.asarray([mu(int(j)) for j in occupancy] if callable(mu) else [mu] * self.k, dtype=float)
        if values.shape != (self.k,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("all per-request service rates must be positive and finite")
        with np.errstate(over="ignore"):
            rates = occupancy * values
        if not np.all(np.isfinite(rates)):
            raise ValueError("total completion rates overflow; rescale the time unit")
        self.departure_rates = np.r_[0.0, rates, np.full(self.truncation - self.k, rates[-1])]
        self.is_servers_set = True

    def _initial(self, initial):
        probabilities = np.zeros(self.truncation + 2)
        if isinstance(initial, Integral) and not isinstance(initial, bool):
            if not 0 <= initial <= self.truncation:
                raise ValueError("initial population must be in 0..queue_truncation")
            probabilities[int(initial)] = 1
        else:
            values = np.asarray(initial, dtype=float)
            if values.shape != (self.truncation + 1,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("initial distribution must be a nonnegative finite vector of length N+1")
            if not math.isclose(float(values.sum()), 1.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError("initial distribution must sum to one")
            probabilities[:-1] = values
        return probabilities

    def _generator(self, rate):
        size = self.truncation + 2
        return diags(
            [np.r_[self.departure_rates[1:], 0.0], np.r_[-rate - self.departure_rates, 0.0], np.full(size - 1, rate)],
            [-1, 0, 1],
            shape=(size, size),
            format="csr",
        )

    def run(self, times, initial=0):
        """Propagate from time zero and return results at increasing sample times.

        ``initial`` is an integer population or a normalized vector on 0..N.
        No stationary warmup or stability requirement is imposed. Scheduling
        changes split propagation intervals even when absent from ``times``.
        """
        self._check_if_servers_and_sources_set()
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("times must be a nonempty finite vector")
        if times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be nonnegative and strictly increasing")
        started = self._measure_time()
        probabilities = self._initial(initial)
        snapshots = []
        now, segment = 0.0, 0
        matrix = self._generator(self.schedule[0, 1])
        for target in times:
            while now < target:
                change = self.schedule[segment + 1, 0] if segment + 1 < len(self.schedule) else math.inf
                stop = min(target, change)
                delta = stop - now
                with np.errstate(over="ignore"):
                    operator = matrix.T * delta
                if not np.all(np.isfinite(operator.data)):
                    raise ValueError("rate-time product overflow; rescale the input")
                probabilities = expm_multiply(operator, probabilities, traceA=float(operator.diagonal().sum()))
                now = stop
                if now == change:
                    segment += 1
                    matrix = self._generator(self.schedule[segment, 1])
            snapshots.append(probabilities.copy())
        snapshots = np.asarray(snapshots)
        residual = float(np.max(np.abs(snapshots.sum(axis=1) - 1)))
        if not np.all(np.isfinite(snapshots)) or snapshots.min() < -1e-12 or residual > 1e-8:
            raise ArithmeticError("matrix-exponential propagation lost probability accuracy")
        result = TransientOccupancyResults(
            times=times.copy(),
            state_probabilities=snapshots[:, :-1],
            escape_probability=snapshots[:, -1],
            arrival_rates=self.schedule[np.searchsorted(self.schedule[:, 0], times, side="right") - 1, 1],
            concurrency_cap=self.k,
            saturated_departure_rate=float(self.departure_rates[-1]),
            mass_residual=residual,
        )
        self._set_duration(result, started)
        return result
