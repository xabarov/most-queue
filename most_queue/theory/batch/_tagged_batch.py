"""Shared tagged-customer PH representation for the three bulk-service solvers.

This is a numerical representation of an infinite waiting room, not a finite
buffer queue. Stationary truncation remains the caller's separate error source.
For positive patience rates, the behind counter has a probabilistic error bound;
see ``_tagged_boundary``. No machine-rounding enclosure is claimed.
"""

import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, splu

from most_queue.theory.batch._tagged_boundary import behind_cap, behind_error


class TaggedBatchWait:
    """One sparse chain and PASTA mixture, reused across moments and deadlines.

    ``calc`` is a configured BulkServiceMM1/Erlang/H2Calc. Public diagnostics
    distinguish omitted initial mass from the behind-counter bound. At gamma=0
    neither initial states nor the (exactly lumpable) behind count are discarded.
    """

    def __init__(self, calc, tolerance=1e-13, max_states=300_000):
        calc._check_if_servers_and_sources_set()
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("tolerance must be finite and in (0, 1)")
        a, b, lam, gamma = calc.a, calc.b, calc.l, calc.gamma
        self.gamma = gamma
        self._arrival_rate = lam
        self.cap = behind_cap(a, lam, gamma, tolerance)
        self.bound = 0.0 if a == 1 or gamma == 0 else behind_error(lam, gamma, self.cap)[0]
        pi = calc._solve_pi()
        qmass = pi.reshape(-1, calc.N + 1).sum(axis=0)
        limit = calc.N
        self.omitted_mass = 0.0
        if gamma > 0:
            while limit > a - 1 and self.omitted_mass + qmass[limit] <= tolerance:
                self.omitted_mass += float(qmass[limit])
                limit -= 1
        self.queue_limit = limit
        self.boundary_mass = float(qmass[-1])
        width = self.cap + 1
        phases, initial, exits, moves = [], {}, {}, {}
        for size in range(a, b + 1):
            if hasattr(calc, "p1_fn"):
                rates = (calc.mu1_fn(size), calc.mu2_fn(size))
                probs = (calc.p1_fn(size), 1 - calc.p1_fn(size))
                for phase, rate in enumerate(rates):
                    phases.append((size, phase))
                    exits[size, phase] = rate
                    moves[size, phase] = None
                initial[size] = probs
            else:
                count = getattr(calc, "k", 1)
                rate = calc.rate_fn(size) if hasattr(calc, "rate_fn") else calc.mu_fn(size)
                initial[size] = (1.0,) + (0.0,) * (count - 1)
                for phase in range(count):
                    phases.append((size, phase))
                    exits[size, phase] = rate if phase == count - 1 else 0.0
                    moves[size, phase] = (phase + 1, rate) if phase < count - 1 else None
        phase_index = {pair: idx for idx, pair in enumerate(phases)}
        if hasattr(calc, "p1_fn"):
            mean_full = calc.p1_fn(b) / calc.mu1_fn(b) + (1 - calc.p1_fn(b)) / calc.mu2_fn(b)
        else:
            mean_full = getattr(calc, "k", 1) / (calc.rate_fn(b) if hasattr(calc, "rate_fn") else calc.mu_fn(b))
        if gamma == 0 and lam * mean_full >= b:
            raise ValueError("stationary infinite queue requires lambda*E[S_b] < b without abandonment")
        block = len(phases) * width
        idle_count = a * (a - 1) // 2
        total = (limit + 1) * block + idle_count
        if total > max_states:
            raise ValueError(f"tagged chain requires {total} states, exceeds budget {max_states}")
        # Idle states come first; busy dependencies either stay in a block R or
        # go to smaller R / idle. This permits block-forward linear solves.
        idle = {(r, k): idx for idx, (r, k) in enumerate((r, k) for r in range(a - 1) for k in range(a - 1 - r))}

        def busy(r, size, phase, k):
            return idle_count + r * block + phase_index[size, phase] * width + k

        rows, cols, values = [], [], []
        diagonal = np.full(total, gamma, dtype=float)
        self.serve_rate = np.zeros(total)
        self.alpha = np.zeros(total)
        self.atom = float(pi[a - 1])

        def add(src, dst, rate):
            if rate:
                rows.append(src)
                cols.append(dst)
                values.append(rate)
                diagonal[src] += rate

        def absorb(src, rate):
            diagonal[src] += rate
            self.serve_rate[src] += rate

        for (r, k), src in idle.items():
            if r:
                add(src, idle[r - 1, k], r * gamma)
            if k:
                add(src, idle[r, k - 1], k * gamma)
            if r + k + 2 >= a:
                absorb(src, lam)
            else:
                add(src, idle[r, k + 1], lam)
            if k == 0:
                self.alpha[src] = pi[r]
        for r in range(limit + 1):
            for size, phase in phases:
                prob_idx = calc._busy_index(size, phase, r) if hasattr(calc, "_busy_index") else calc._index(size, r)
                self.alpha[busy(r, size, phase, 0)] = pi[prob_idx]
                for k in range(width):
                    src = busy(r, size, phase, k)
                    if r:
                        add(src, busy(r - 1, size, phase, k), r * gamma)
                    if k and gamma:
                        add(src, busy(r, size, phase, k - 1), k * gamma)
                    if k < self.cap:
                        add(src, busy(r, size, phase, k + 1), lam)
                    move = moves[size, phase]
                    if move is not None:
                        add(src, busy(r, size, move[0], k), move[1])
                    rate = exits[size, phase]
                    if r >= b:
                        for next_phase, probability in enumerate(initial[b]):
                            add(src, busy(r - b, b, next_phase, k), rate * probability)
                    elif r + k + 1 >= a:
                        absorb(src, rate)
                    else:
                        add(src, idle[r, k], rate)
        self.sub = (sp.coo_matrix((values, (rows, cols)), shape=(total, total)).tocsr() - sp.diags(diagonal)).tocsr()
        self._blocks = []
        neg = -self.sub
        intervals = ([(0, idle_count)] if idle_count else []) + [
            (idle_count + r * block, idle_count + (r + 1) * block) for r in range(limit + 1)
        ]
        for start, stop in intervals:
            self._blocks.append(
                (start, stop, splu(neg[start:stop, start:stop].tocsc()), neg[start:stop, :start].tocsr())
            )
        self.h = self.solve(self.serve_rate)
        self.served = float(self.atom + self.alpha @ self.h)
        self.abandoned = float(gamma * (self.alpha @ self.solve(np.ones(total))))

    def solve(self, rhs):
        """Solve (-T)x=rhs using the non-increasing number ahead of the tag."""
        out = np.zeros_like(rhs, dtype=float)
        for start, stop, factor, lower in self._blocks:
            out[start:stop] = factor.solve(rhs[start:stop] - lower @ out[:start])
        return out

    def moments(self, num=4):
        """Raw waiting moments conditional on dispatch, including its zero atom."""
        if not isinstance(num, int) or num < 1:
            raise ValueError("num must be a positive integer")
        x = self.h.copy()
        result = []
        for order in range(1, num + 1):
            x = self.solve(x)
            result.append(float(math.factorial(order) * (self.alpha @ x) / self.served))
        return result

    def tail(self, time):
        """P(W>time | dispatched). Negative deadlines include the zero atom."""
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        if time < 0:
            return 1.0
        value = self.h if time == 0 else expm_multiply(self.sub * time, self.h)
        return float(self.alpha @ value / self.served)

    def outcomes(self, deadline):
        """On-time, late dispatch, abandonment (discarded mass kept separate)."""
        late = self.served * self.tail(deadline)
        return {
            "on_time": self.served - late,
            "late": late,
            "abandoned": self.abandoned,
            "omitted_mass": self.omitted_mass,
            "behind_error_bound": self.bound,
        }

    def moment_error_bounds(self, num=4):
        """Absolute local bounds for *unnormalized* dispatched-wait moments.

        These exclude stationary truncation and floating-point evaluation error.
        """
        if self.gamma == 0:
            return [0.0] * (num + 1)
        bound = (0.0,) * (num + 1) if self.cap == 0 else behind_error(self._arrival_rate, self.gamma, self.cap, num)
        return [bound[n] + self.omitted_mass * math.factorial(n) / self.gamma**n for n in range(num + 1)]


def tagged_wait(calc):
    """Cache a configured solver's representation; setters invalidate this cache."""
    result = getattr(calc, "_tagged_wait", None)
    if result is None:
        result = TaggedBatchWait(calc)
        calc._tagged_wait = result
    return result
