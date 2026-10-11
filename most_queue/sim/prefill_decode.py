"""Independent iteration-level prefill/decode DES; no analytical rates/rewards.

Uses the stage costs of Ramani and Tantawi, arXiv:2609.20957v1, but follows
individual integer tokens. At each boundary admit FCFS up to B; run a decode
token for every decoding request, then spend the remaining budget on FCFS
prefills. Prefill completion never decodes in the SAME iteration. Arrivals
during execution can join only at the next boundary. This specified policy
is not a reproduction of vLLM or of the paper's mean-field scheduler.
"""

# Costs/lengths match the documented mathematical model, not arbitrary optional settings.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import math
import time
from collections import deque

import numpy as np

from most_queue.structs import PrefillDecodeSimResults


class PrefillDecodeSim:
    """Token-by-token continuous batching with deterministic stage durations.

    run() generates a finite Poisson arrival cohort then drains it. It does
    not certify steady-state stability: overloaded traces can finish too.
    replay() accepts individual integer lengths and an ordered arrival trace.
    """

    def __init__(self, batch_cap=256, token_budget=8192, seed=None, max_iterations=2_000_000):
        if isinstance(batch_cap, bool) or not isinstance(batch_cap, int) or batch_cap < 1:
            raise ValueError("batch_cap must be a positive integer")
        if token_budget is not None and (
            isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < batch_cap
        ):
            raise ValueError("token_budget must be None or an integer >= batch_cap")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.batch_cap, self.token_budget = batch_cap, token_budget
        self.seed, self.max_iterations = seed, max_iterations
        self.l = None
        self.alpha = self.beta = self.gamma = None
        self.input_tokens = self.output_tokens = None
        self.length_sampler = None

    def set_sources(self, l):
        """Set the finite positive Poisson arrival rate."""
        if not np.isfinite(l) or l <= 0:
            raise ValueError("arrival rate must be positive and finite")
        self.l = float(l)

    def set_servers(self, input_tokens, output_tokens, alpha, beta, gamma, *, length_sampler=None):
        """Set constant lengths/costs or sampler(rng)->(input_length, output_length)."""
        self._lengths([input_tokens], [output_tokens])
        costs = np.asarray([alpha, beta, gamma], dtype=float)
        if not np.all(np.isfinite(costs)) or costs[0] <= 0 or np.any(costs[1:] < 0):
            raise ValueError("alpha must be positive, beta/gamma nonnegative, all finite")
        if length_sampler is not None and not callable(length_sampler):
            raise ValueError("length_sampler must be callable or None")
        self.input_tokens, self.output_tokens = int(input_tokens), int(output_tokens)
        self.alpha, self.beta, self.gamma = costs.tolist()
        self.length_sampler = length_sampler

    @staticmethod
    def _lengths(inputs, outputs):
        for values in (inputs, outputs):
            a = np.asarray(values)
            if a.ndim != 1 or not np.all(np.isfinite(a)) or np.any(a <= 0) or np.any(a != np.floor(a)):
                raise ValueError("individual token lengths must be positive finite integers")
            if np.any(a > 1_000_000):
                raise ValueError("individual lengths exceed the 1000000-token safety budget")

    def run(self, num_of_jobs=100_000, warmup_fraction=0.2, *, record_iterations=False):
        """Generate a finite arrival cohort and return post-warmup latency statistics."""
        if self.l is None or self.alpha is None:
            raise ValueError("set_sources and set_servers must be called before run")
        if isinstance(num_of_jobs, bool) or not isinstance(num_of_jobs, int) or num_of_jobs < 1:
            raise ValueError("num_of_jobs must be a positive integer")
        rng = np.random.default_rng(self.seed)
        arrivals = np.cumsum(rng.exponential(1 / self.l, num_of_jobs))
        if self.length_sampler is None:
            inputs = np.full(num_of_jobs, self.input_tokens, dtype=int)
            outputs = np.full(num_of_jobs, self.output_tokens, dtype=int)
        else:
            lengths = [self.length_sampler(rng) for _ in range(num_of_jobs)]
            inputs, outputs = np.asarray(lengths).T
        return self.replay(arrivals, inputs, outputs, warmup_fraction, record_iterations=record_iterations)

    def replay(self, arrivals, inputs, outputs, warmup_fraction=0.0, *, record_iterations=False):
        """Replay an ordered trace, including ties, starting from an empty server.

        Start times mark admission; prefill_mean includes budget stalls after
        admission. Occupancy is observed from the first measured arrival to
        the last input arrival, and is not distorted by the final drain.
        Full timestamp arrays include warmup jobs. Iteration recording is
        opt-in and bounded by max_iterations.
        """
        if self.alpha is None:
            raise ValueError("set_servers must be called before replay")
        arrivals = np.asarray(arrivals, dtype=float)
        if arrivals.ndim != 1 or arrivals.size == 0 or not np.all(np.isfinite(arrivals)):
            raise ValueError("arrivals must be a nonempty finite vector")
        if arrivals[0] < 0 or np.any(np.diff(arrivals) < 0):
            raise ValueError("arrivals must be nonnegative and nondecreasing")
        self._lengths(inputs, outputs)
        if len(inputs) != len(arrivals) or len(outputs) != len(arrivals):
            raise ValueError("arrival and length vectors must have the same size")
        if not np.isfinite(warmup_fraction) or not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0,1)")
        started = time.process_time()
        inputs, outputs = np.asarray(inputs, dtype=int), np.asarray(outputs, dtype=int)
        count = len(arrivals)
        starts, pref_end, completes = np.zeros(count), np.zeros(count), np.zeros(count)
        processed = np.zeros(count, dtype=int)
        tokens = [[] for _ in range(count)]
        queued, active = deque(), []
        cursor = 0
        now = 0.0
        warmup = int(count * warmup_fraction)
        begin, end = arrivals[warmup], arrivals[-1]
        state_area = {}
        active_area = busy_area = 0.0
        trace = []

        def advance(target):
            nonlocal now, active_area, busy_area
            dt = max(0.0, min(target, end) - max(now, begin))
            state = len(active) + len(queued)
            if dt > 0:
                state_area[state] = state_area.get(state, 0.0) + dt
            active_area += dt * len(active)
            busy_area += dt * bool(active)
            now = target

        def accept_until(boundary):
            nonlocal cursor
            while cursor < count and arrivals[cursor] <= boundary:
                advance(float(arrivals[cursor]))
                queued.append(cursor)
                cursor += 1
            advance(boundary)

        iterations = 0
        while cursor < count or active or queued:
            if not active and not queued:
                accept_until(float(arrivals[cursor]))
            while queued and len(active) < self.batch_cap:
                job = queued.popleft()
                starts[job] = now
                active.append(job)
            decoders = [j for j in active if processed[j] >= inputs[j]]
            prefills = [j for j in active if processed[j] < inputs[j]]
            remaining = math.inf if self.token_budget is None else self.token_budget - len(decoders)
            work = self.alpha
            actions = []
            # A decoder uses exactly one token and reads its entire new cache.
            for job in decoders:
                cache = processed[job] + 1
                work += self.beta + self.gamma * cache
                actions.append((job, "decode", 1))
            # Admitted prefills retain FCFS order even when temporarily stalled.
            for job in prefills:
                amount = int(min(inputs[job] - processed[job], remaining))
                if amount <= 0:
                    continue
                work += self.beta * amount + self.gamma * (processed[job] + amount)
                remaining -= amount
                actions.append((job, "prefill", amount))
            finish = now + work
            if not math.isfinite(finish) or finish <= now:
                raise ValueError("iteration duration overflowed or failed to advance the clock")
            iterations += 1
            if iterations > self.max_iterations:
                raise RuntimeError("max_iterations exhausted; increase the explicit simulation budget")
            if record_iterations:
                trace.append(
                    {
                        "start": now,
                        "end": finish,
                        "actions": actions,
                        "tokens": sum(a[2] for a in actions),
                        "active": active.copy(),
                    }
                )
            accept_until(finish)
            departed = set()
            for job, stage, amount in actions:
                processed[job] += amount
                if stage == "prefill":
                    if processed[job] == inputs[job]:
                        pref_end[job] = finish
                else:
                    tokens[job].append(finish)
                    if len(tokens[job]) == outputs[job]:
                        completes[job] = finish
                        departed.add(job)
            active = [job for job in active if job not in departed]

        waits = starts[warmup:] - arrivals[warmup:]
        sojourns = completes[warmup:] - arrivals[warmup:]
        first = np.array([t[0] for t in tokens[warmup:]])
        first_gaps = first - pref_end[warmup:]
        actual_gaps = [b - a for row in tokens[warmup:] for a, b in zip(row, row[1:])]
        decode_total = sum(completes[warmup:] - pref_end[warmup:])
        observation = end - begin
        pmf = [state_area.get(i, 0) / observation for i in range(max(state_area) + 1)] if observation else None
        jobs_mean = sum(i * area for i, area in state_area.items()) / observation if observation else 0.0
        batch_mean = active_area / observation if observation else 0.0
        busy = busy_area / observation if observation else 0.0
        return PrefillDecodeSimResults(
            w=[float(np.mean(waits**r)) for r in range(1, 4)],
            v=[float(np.mean(sojourns**r)) for r in range(1, 4)],
            p=pmf,
            utilization=busy,
            ttft_mean=float(np.mean(first - arrivals[warmup:])),
            itl_mean=float(np.mean(actual_gaps)) if actual_gaps else None,
            prefill_mean=float(np.mean(pref_end[warmup:] - starts[warmup:])),
            first_decode_mean=float(np.mean(first_gaps)),
            all_decode_mean=float(decode_total / sum(outputs[warmup:])),
            service_mean=float(np.mean(completes[warmup:] - starts[warmup:])),
            batch_mean=batch_mean,
            jobs_mean=jobs_mean,
            queue_mean=jobs_mean - batch_mean,
            busy_probability=busy,
            waiting_probability=float(np.mean(waits > 0)),
            observation_time=float(observation),
            start_times=starts.tolist(),
            prefill_end_times=pref_end.tolist(),
            completion_times=completes.tolist(),
            output_times=tokens,
            iteration_trace=trace,
            duration=time.process_time() - started,
        )
