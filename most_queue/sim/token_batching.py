"""Independent request/iteration DES for Yang--Jiao--Xu (2024) batching.

Actual iid token lengths are sampled at arrival. Elastic completion offsets
are constructed by shrinking the active set at token-level boundaries, not
by sampling the analytic maximum/sum distribution or importing its rewards.
"""

import time
from collections import deque
from numbers import Integral

import numpy as np

from most_queue.sim.dynamic_batching import DynamicBatchingSimResults
from most_queue.theory.batch.token_service import TokenLengthDistribution


class TokenBatchingSim:
    """Dynamic, fixed, and gated elastic service; optional physical dynamic cap."""

    def __init__(self, strategy="dynamic", batch_size=None, seed=None):
        if strategy not in ("dynamic", "fixed", "elastic"):
            raise ValueError("strategy must be dynamic, fixed or elastic")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if strategy == "fixed" and batch_size is None:
            raise ValueError("fixed strategy requires batch_size")
        self.strategy, self.batch_size = strategy, batch_size
        self.generator = np.random.default_rng(seed)
        self.rate = self.tokens = None
        self.k1 = self.k2 = self.k3 = self.k4 = 0.0

    def set_sources(self, rate):
        """Set positive finite Poisson arrival rate."""
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be finite and positive")
        self.rate = float(rate)
        return self

    def set_servers(self, tokens, k1, k2, k3, k4):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Set total-token PMF and positive prefill/nonnegative decode costs."""
        coefficients = np.asarray([k1, k2, k3, k4], dtype=float)
        if not isinstance(tokens, TokenLengthDistribution):
            raise ValueError("tokens must be a TokenLengthDistribution")
        if not np.all(np.isfinite(coefficients)) or k1 <= 0 or np.any(coefficients < 0):
            raise ValueError("require finite k1>0 and k2/k3/k4>=0")
        self.tokens, self.k1, self.k2, self.k3, self.k4 = tokens, *coefficients
        return self

    def completion_offsets(self, total_lengths):
        """Compute completion times by explicit active-set token iterations.

        Public for deterministic trace checks; no stochastic service law
        or theoretical conditional moments are consulted.
        """
        n = np.asarray(total_lengths, dtype=float) - 1
        if n.ndim != 1 or n.size == 0 or not np.all(np.isfinite(n)) or np.any(n < 0) or np.any(n != np.floor(n)):
            raise ValueError("lengths must be a nonempty vector of positive integer totals")
        pre = self.k1 * n.size + self.k2
        if self.strategy != "elastic":
            return np.full(n.size, pre + (self.k3 * n.size + self.k4) * n.max())
        offsets = np.full(n.size, pre)
        active, previous, clock = n > 0, 0.0, pre
        for level in np.unique(n[n > 0]):
            clock += (level - previous) * (self.k3 * int(active.sum()) + self.k4)
            done = active & (n == level)
            offsets[done] = clock
            active[done] = False
            previous = level
        return offsets

    def _check_ready(self):
        if self.rate is None or self.tokens is None:
            raise ValueError("set_sources/set_servers must be called first")
        n, p = self.tokens.lengths - 1, self.tokens.probabilities
        if self.batch_size is None:
            if self.rate * (self.k1 + self.k3 * n[-1]) >= 1:
                raise ValueError("unbounded simulation requires lambda*(k1+k3*max_decode)<1")
        else:
            b = self.batch_size
            maxima = np.diff(np.concatenate(([0.0], np.cumsum(p) ** b)))
            if self.strategy == "elastic":
                mean = self.k1 * b + self.k2 + self.k3 * b * float(p @ n) + self.k4 * float(maxima @ n)
            else:
                mean = self.k1 * b + self.k2 + (self.k3 * b + self.k4) * float(maxima @ n)
            if self.rate * mean >= b:
                raise ValueError("fixed/capped simulation requires lambda*E[H|cap]<cap")

    def run(self, num_of_jobs=100_000, warmup_fraction=0.2):
        # pylint: disable=too-many-locals,too-many-statements
        """Measure whole completed batches after whole-batch warmup.

        Occupancy/utilization include actual early releases. Last batch can
        overshoot num_of_jobs. A fixed warmup fraction is not a guarantee
        of stationarity; compare independent seeds and larger budgets.
        """
        self._check_ready()
        if isinstance(num_of_jobs, bool) or not isinstance(num_of_jobs, Integral) or num_of_jobs < 1:
            raise ValueError("num_of_jobs must be a positive integer")
        if not np.isfinite(warmup_fraction) or not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must lie in [0,1)")
        start = time.process_time()
        target = int(num_of_jobs * warmup_fraction / (1 - warmup_fraction))
        warmup = completed = 0
        clock = observation_start = busy_time = 0.0
        next_arrival = float(self.generator.exponential(1 / self.rate))
        queue, state_times = deque(), {}
        waits, latencies, sizes = [], [], []

        def add_time(end, active, measuring):
            nonlocal clock
            if measuring:
                count = active + len(queue)
                state_times[count] = state_times.get(count, 0.0) + end - clock
            clock = end

        def arrive(active, measuring):
            nonlocal next_arrival
            add_time(next_arrival, active, measuring)
            length = float(self.tokens.sample(self.generator, 1)[0])
            queue.append((clock, length))
            increment = float(self.generator.exponential(1 / self.rate))
            next_arrival = clock + increment
            if not np.isfinite(next_arrival) or next_arrival <= clock:
                raise RuntimeError("arrival time cannot advance the simulation clock")

        while len(waits) < num_of_jobs:
            measuring = completed >= target
            required = self.batch_size if self.strategy == "fixed" else 1
            while len(queue) < required:
                arrive(0, measuring)
            size = len(queue) if self.batch_size is None else min(len(queue), self.batch_size)
            batch = [queue.popleft() for _ in range(size)]
            arrivals, lengths = np.asarray(batch).T
            begin = clock
            offsets = self.completion_offsets(lengths)
            active = size
            for offset in np.unique(offsets):
                finish = begin + offset
                if not np.isfinite(finish) or finish <= begin:
                    raise RuntimeError("service time cannot advance the simulation clock")
                while next_arrival <= finish:
                    arrive(active, measuring)
                add_time(finish, active, measuring)
                active -= int((offsets == offset).sum())
            if measuring:
                waits.extend(begin - arrivals)
                latencies.extend(begin + offsets - arrivals)
                sizes.append(size)
                busy_time += clock - begin
            else:
                warmup += size
                observation_start = clock
            completed += size
        w, v = np.asarray(waits), np.asarray(latencies)
        elapsed = clock - observation_start
        p = np.zeros(max(state_times) + 1)
        for count, duration in state_times.items():
            p[count] = duration / elapsed
        utilization = busy_time / elapsed
        return DynamicBatchingSimResults(
            w=[float(np.mean(w**order)) for order in (1, 2, 3)],
            v=[float(np.mean(v**order)) for order in (1, 2, 3)],
            p=p.tolist(),
            utilization=utilization,
            waiting_mean=float(w.mean()),
            latency_mean=float(v.mean()),
            service_mean=float((v - w).mean()),
            batch_mean=float(np.mean(sizes)),
            idle_fraction=1 - utilization,
            served=w.size,
            batches=len(sizes),
            warmup_served=warmup,
            observation_time=elapsed,
            waits=w,
            latencies=v,
            duration=time.process_time() - start,
        )
