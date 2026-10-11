"""Exact heterogeneous exponential bulk service (van Ommeren et al., 2020).

DOI: 10.1007/s11134-020-09654-y, §§3–4, Theorem 2 and Remark 1.
The queue is unbounded: only the free-period and waiting PH chains are finite.
Printed coefficient/index typos are resolved by the physical event rules;
see docs/research/heterogeneous-bulk-service-validation-2026.md.
This is NOT general service, continuous batching, or a token/KV model.
"""

import itertools
import math
import numbers

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import expm_multiply, spsolve

from most_queue.structs import HeterogeneousBulkResults
from most_queue.theory.base_queue import BaseQueue


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a positive finite real")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real")
    return value


class BulkServiceHeterogeneousCalc(BaseQueue):
    """Poisson/FCFS, type-specific counts, rates, thresholds and capacities.

    Types are ordered by threshold, then input order (tie priority).
    Result type arrays always use input order. Resource limits: 20,000 free
    states including the busy aggregate, 1,500 waiting phases, B <= 1,024.
    These limits concern computation, not a finite physical waiting room.
    """

    def __init__(self, counts):
        counts = tuple(_integer(x, "count") for x in counts)
        if not counts or any(x < 1 for x in counts):
            raise ValueError("counts must be a nonempty sequence of positive integers")
        super().__init__(sum(counts))
        self.counts = counts
        self.l = None
        self.rates = None
        self.thresholds = None
        self.capacities = None
        self._result = None
        self.free_states = []
        self.free_generator = None
        self.waiting_states = []
        self.waiting_generator = None
        self.waiting_initial = None
        self.waiting_absorption = None
        self.response_generator = None
        self.response_initial = None

    def _invalidate(self):
        self._result = None
        self.w = self.v = self.ro = None
        self.mean_jobs_on_queue = self.mean_jobs_in_system = None
        self.free_states = []
        self.waiting_states = []
        self.free_generator = self.waiting_generator = self.response_generator = None
        self.waiting_initial = self.waiting_absorption = self.response_initial = None

    def set_sources(self, l):
        """Set positive Poisson arrival rate and invalidate previous results."""
        self.l = _positive(l, "arrival rate")
        self.is_sources_set = True
        self._invalidate()

    def set_servers(self, rates, thresholds, capacities):
        """Batch duration is Exp(rate), independent of the selected batch size."""
        rates = tuple(_positive(x, "rate") for x in rates)
        thresholds = tuple(_integer(x, "threshold") for x in thresholds)
        capacities = tuple(_integer(x, "capacity") for x in capacities)
        if not len(rates) == len(thresholds) == len(capacities) == len(self.counts):
            raise ValueError("type arrays must have the same length as counts")
        if any(not 1 <= t <= b <= 1024 for t, b in zip(thresholds, capacities)):
            raise ValueError("require 1 <= threshold <= capacity <= 1024")
        if max(thresholds) + sum(t * (t - 1) // 2 for t in thresholds) > 1500:
            raise ValueError("waiting PH exceeds 1500 phases")
        # Count states without materializing the potentially enormous product.
        ordered = sorted(range(len(rates)), key=lambda i: (thresholds[i], i))
        free_count = 1 + sum(
            thresholds[i] * self.counts[i] * math.prod(self.counts[j] + 1 for j in ordered[k + 1 :])
            for k, i in enumerate(ordered)
        )
        if free_count > 20000:
            raise ValueError("free-period chain exceeds 20000 states")
        self.rates, self.thresholds, self.capacities = rates, thresholds, capacities
        self.is_servers_set = True
        self._invalidate()

    @staticmethod
    def _generator(rows, size):
        """Assemble off-diagonal rate dictionaries, omitting self events."""
        rr, cc, vv = [], [], []
        for i, row in enumerate(rows):
            total = 0.0
            for j, rate in row.items():
                if j != i and rate:
                    rr.append(i)
                    cc.append(j)
                    vv.append(rate)
                    total += rate
            rr.append(i)
            cc.append(i)
            vv.append(-total)
        return csr_matrix((vv, (rr, cc)), shape=(size, size))

    def _free_chain(self, order, alpha):
        states = []
        for k, sigma in enumerate(order):
            ranges = [range(self.counts[i], self.counts[i] + 1) for i in order[:k]]
            ranges += [range(self.counts[sigma])]
            ranges += [range(self.counts[i] + 1) for i in order[k + 1 :]]
            for vector in itertools.product(*ranges):
                busy = tuple(vector[order.index(i)] for i in range(len(order)))
                states.extend((busy, q) for q in range(self.thresholds[sigma]))
        bp = len(states)
        index = {s: i for i, s in enumerate(states)}
        rows = [{} for _ in range(bp + 1)]
        flow = np.zeros(len(order))
        starts = np.zeros(len(order))
        atom = np.zeros(len(order))
        init = []
        for idx, (busy, q) in enumerate(states):
            sigma = next(i for i in order if busy[i] < self.counts[i])
            new = list(busy)
            if q + 1 == self.thresholds[sigma]:
                new[sigma] += 1
                target = bp if tuple(new) == self.counts else index[(tuple(new), 0)]
                init.append((idx, sigma, None))
            else:
                target = index[(busy, q + 1)]
                init.append((idx, sigma, q + 1))
            rows[idx][target] = rows[idx].get(target, 0.0) + self.l
            for i, count in enumerate(busy):
                if not count:
                    continue
                new = list(busy)
                if q < self.thresholds[i]:
                    new[i] -= 1
                    target = index[(tuple(new), q)]
                else:
                    target = index[(busy, max(0, q - self.capacities[i]))]
                rows[idx][target] = rows[idx].get(target, 0.0) + count * self.rates[i]
        for i, count in enumerate(self.counts):
            new = list(self.counts)
            new[i] -= 1
            for q in range(self.thresholds[i]):
                rows[bp][index[(tuple(new), q)]] = count * self.rates[i] * (1 - alpha) * alpha**q
        generator = self._generator(rows, bp + 1)
        scaled = generator / max(self.l, max(n * mu for n, mu in zip(self.counts, self.rates)))
        equation = scaled.T.tolil()
        equation[-1, :] = np.ones(bp + 1)
        rhs = np.zeros(bp + 1)
        rhs[-1] = 1
        mass = spsolve(equation.tocsr(), rhs)
        if not np.all(np.isfinite(mass)) or mass.min() < -1e-10:
            raise ArithmeticError("stationary solve failed")
        mass = np.maximum(mass, 0)
        mass /= mass.sum()
        residual = float(np.max(np.abs(mass @ scaled)))
        for idx, sigma, m in init:
            if m is None:
                rate = self.l * mass[idx]
                atom[sigma] += mass[idx]
                starts[sigma] += rate
                flow[sigma] += self.thresholds[sigma] * rate
        for idx, (busy, q) in enumerate(states):
            for i, count in enumerate(busy):
                if q >= self.thresholds[i]:
                    rate = count * self.rates[i] * mass[idx]
                    starts[i] += rate
                    flow[i] += min(q, self.capacities[i]) * rate
        for i, (n, mu, t, b) in enumerate(zip(self.counts, self.rates, self.thresholds, self.capacities)):
            rate = mass[bp] * n * mu
            starts[i] += rate * alpha**t
            flow[i] += rate * (b * alpha**b + (1 - alpha) * sum(q * alpha**q for q in range(t, b)))
        self.free_states = states
        self.free_generator = generator
        return mass, flow, starts, atom, init, residual

    def _waiting_chain(
        self, order, alpha, mass, atom, init
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        limit = max(self.thresholds) - 1
        states = [(None, 0, ell) for ell in range(limit + 1)]
        states += [
            (sigma, m, ell)
            for sigma in order
            for ell in range(self.thresholds[sigma] - 1)
            for m in range(1, self.thresholds[sigma] - ell)
        ]
        index = {s: i for i, s in enumerate(states)}
        rows = [{} for _ in states]
        exits = np.zeros((len(states), len(order)))
        initial = np.zeros(len(states))
        initial[0] = mass[-1]
        for idx, sigma, m in init:
            if m is not None:
                initial[index[(sigma, m, 0)]] += mass[idx]
        for idx, (sigma, m, ell) in enumerate(states):
            if sigma is None:
                if ell < limit:
                    rows[idx][index[(None, 0, ell + 1)]] = self.l
                for i, (n, mu, t, b) in enumerate(zip(self.counts, self.rates, self.thresholds, self.capacities)):
                    for pos in range(1, t - ell):
                        rows[idx][index[(i, pos, ell)]] = n * mu * (1 - alpha) * alpha ** (pos - 1)
                    exits[idx, i] = n * mu * (alpha ** max(t - ell - 1, 0) - alpha**b)
            else:
                if m + ell + 1 == self.thresholds[sigma]:
                    exits[idx, sigma] += self.l
                else:
                    rows[idx][index[(sigma, m, ell + 1)]] = self.l
                for i in order[: order.index(sigma)]:
                    rate = self.counts[i] * self.rates[i]
                    if m + ell < self.thresholds[i]:
                        target = index[(i, m, ell)]
                    elif m <= self.capacities[i]:
                        exits[idx, i] += rate
                        continue
                    else:
                        # The completing type immediately restarts; sigma remains idle.
                        target = index[(sigma, m - self.capacities[i], ell)]
                    rows[idx][target] = rows[idx].get(target, 0.0) + rate
        transient = self._generator(rows, len(states)) - diags(exits.sum(axis=1))
        self.waiting_states = states
        self.waiting_generator = transient
        self.waiting_initial = initial
        self.waiting_absorption = exits
        self.response_generator = bmat([[transient, csr_matrix(exits)], [None, -diags(self.rates)]], format="csr")
        self.response_initial = np.concatenate([initial, atom])

    @staticmethod
    def _moments(generator, initial, count):
        vector = np.ones(generator.shape[0])
        result = []
        for k in range(1, count + 1):
            vector = spsolve(-generator, vector)
            result.append(float(math.factorial(k) * (initial @ vector)))
        return result

    def run(self, num=3):
        """Return raw latency moments and exact stationary queue diagnostics."""
        self._check_if_servers_and_sources_set()
        num = _integer(num, "moment count")
        if not 1 <= num <= 8:
            raise ValueError("moment count must be in 1..8")
        start = self._measure_time()
        capacity = sum(n * mu * b for n, mu, b in zip(self.counts, self.rates, self.capacities))
        if not math.isfinite(capacity):
            raise ValueError("aggregate capacity must be finite")
        self.ro = self.l / capacity
        if self.ro >= 1 - 1e-10:
            raise ValueError("require load < 1 - 1e-10 (stability/numerical boundary guard)")
        order = sorted(range(len(self.counts)), key=lambda i: (self.thresholds[i], i))

        def root(a):
            return (
                sum(
                    n * mu * sum(a**k for k in range(1, b + 1))
                    for n, mu, b in zip(self.counts, self.rates, self.capacities)
                )
                / self.l
                - 1
            )

        alpha = brentq(root, 0.0, 1.0, xtol=5e-324)
        mass, flow, starts, atom, init, residual = self._free_chain(order, alpha)
        self._waiting_chain(order, alpha, mass, atom, init)
        self.w = self._moments(self.waiting_generator, self.waiting_initial, num)
        self.v = self._moments(self.response_generator, self.response_initial, num)
        qmean = sum(p * q for p, (_, q) in zip(mass[:-1], self.free_states)) + mass[-1] * alpha / (1 - alpha)
        busy = [
            sum(p * vec[i] for p, (vec, _) in zip(mass[:-1], self.free_states)) + mass[-1] * n
            for i, n in enumerate(self.counts)
        ]
        served = atom + self.waiting_initial @ spsolve(-self.waiting_generator, self.waiting_absorption)
        d = sum(n * mu * (1 - alpha**t) for n, mu, t in zip(self.counts, self.rates, self.thresholds))
        self.mean_jobs_on_queue = qmean
        self.mean_jobs_in_system = self.l * self.v[0]
        result = HeterogeneousBulkResults(
            w=self.w.copy(),
            v=self.v.copy(),
            utilization=self.ro,
            alpha=alpha,
            busy_period_probability=float(mass[-1]),
            free_probabilities=mass[:-1].tolist(),
            mean_queue=qmean,
            mean_population=self.mean_jobs_in_system,
            zero_wait_probability=float(atom.sum()),
            served_probabilities=served.tolist(),
            customer_throughputs=flow.tolist(),
            batch_throughputs=starts.tolist(),
            mean_busy_servers=busy,
            mean_busy_period=1 / d,
            mean_free_period=(1 - mass[-1]) / (mass[-1] * d),
            root_residual=abs(root(alpha)),
            balance_residual=residual,
            little_residual=abs(qmean - self.l * self.w[0]),
            flow_residual=abs(float(flow.sum()) / self.l - 1),
            class_residual=float(np.max(np.abs(served - flow / self.l))),
            free_state_count=len(mass),
            waiting_phase_count=len(self.waiting_states),
        )
        if max(result.balance_residual, result.flow_residual, result.class_residual) > 1e-7:
            raise ArithmeticError("balance/flow/PH consistency check failed")
        if result.little_residual > 1e-7 * max(1, qmean):
            raise ArithmeticError("Little/PH consistency check failed")
        self._set_duration(result, start)
        self._result = result
        return result

    def _ensure(self):
        if self._result is None:
            self.run()

    def get_w(self):
        self._ensure()
        return self.w.copy()

    def get_v(self):
        self._ensure()
        return self.v.copy()

    def queue_probability(self, q):
        """Exact P(Q=q) for waiting jobs; NOT total system population."""
        q = _integer(q, "queue length")
        if q < 0:
            raise ValueError("queue length must be nonnegative")
        self._ensure()
        r = self._result
        return sum(p for p, (_, value) in zip(r.free_probabilities, self.free_states) if value == q) + (
            r.busy_period_probability * (1 - r.alpha) * r.alpha**q
        )

    def _ph(self, response):
        self._ensure()
        if response:
            return self.response_generator, self.response_initial, 0.0
        return self.waiting_generator, self.waiting_initial, self._result.zero_wait_probability

    def tail(self, t, response=False):
        """Strict P(W>t), or P(V>t); at zero exclude the waiting atom."""
        if isinstance(t, bool) or not isinstance(t, numbers.Real) or not math.isfinite(t) or t < 0:
            raise ValueError("time must be nonnegative and finite")
        generator, initial, _ = self._ph(response)
        return float(np.clip(initial @ expm_multiply(generator * t, np.ones(generator.shape[0])), 0, 1))

    def lst(self, s, response=False):
        """Laplace–Stieltjes transform for a nonnegative finite real argument."""
        if isinstance(s, bool) or not isinstance(s, numbers.Real) or not math.isfinite(s) or s < 0:
            raise ValueError("transform argument must be nonnegative and finite")
        generator, initial, atom = self._ph(response)
        exit_rates = -generator @ np.ones(generator.shape[0])
        return float(atom + initial @ spsolve(diags(np.full(generator.shape[0], s)) - generator, exit_rates))
