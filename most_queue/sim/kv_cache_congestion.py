"""Independent integer-request KV eviction/restart simulation for EPIC-088.

Ao et al., arXiv:2606.15555v1, Section 2 protocol. This FCFS integer process
is NOT the continuous proportional map used by the stability theorems.
No imports from theory: each request carries class, decode age, and identity.
"""

import math
import time
from collections import deque
from numbers import Integral, Real

import numpy as np

from most_queue.structs import KvCacheCongestionSimResults


def _int(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < int(not zero):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return int(value)


class KvCacheCongestionSim:
    """LPF eviction, complete restart at stage zero, FCFS admission.

    Ties in LPF evict the newest request first. No proportional fractional
    trimming, GPU calibration, recovery-time charge, or censored latencies.
    Saturated runs generate fresh requests only when the FCFS queue is empty;
    generated requests are counted explicitly, not as exogenous Poisson input.
    """

    def __init__(self, memory_tokens, seed=42, *, max_decode_length=256):
        self.memory_tokens = _int(memory_tokens, "memory_tokens")
        self.max_decode_length = _int(max_decode_length, "max_decode_length")
        self.seed = _int(seed, "seed", zero=True)
        self.rates = None
        self.inputs = None
        self.decodes = None

    def set_sources(self, rates):
        """Set independent Poisson arrival rates per iteration by class."""
        values = []
        for value in rates:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError("rates must be finite nonnegative reals")
            values.append(float(value))
        if not values or (self.inputs is not None and len(values) != len(self.inputs)):
            raise ValueError("one rate per class is required")
        if math.fsum(values) > 1e6:
            raise ValueError("total arrival rate is too large for request-level simulation")
        self.rates = values

    def set_servers(self, input_tokens, decode_tokens):
        """Set fixed class lengths; reserve the next decoding token in memory."""
        inputs = [_int(x, "input_tokens", zero=True) for x in input_tokens]
        decodes = [_int(x, "decode_tokens") for x in decode_tokens]
        if not inputs or len(inputs) != len(decodes):
            raise ValueError("input/decode vectors must have equal nonzero length")
        if self.rates is not None and len(inputs) != len(self.rates):
            raise ValueError("one rate per class is required")
        if max(decodes) > self.max_decode_length or any(s + d >= self.memory_tokens for s, d in zip(inputs, decodes)):
            raise ValueError("decode length exceeds limit or peak footprint does not fit strictly")
        self.inputs, self.decodes = inputs, decodes

    def run(  # pylint: disable=too-many-arguments
        self,
        iterations,
        *,
        arrivals=None,
        saturated=False,
        initial_state=None,
        initial_queue=None,
        admission_cap=None,
        warmup=0,
        record_trace=False,
    ):
        """Simulate a fixed iteration horizon without draining.

        arrivals is an exact nonnegative integer array [iterations, classes].
        admission_cap is an integer (no silent floor of the continuous cap).
        initial_state gives integer counts by class/stage; initial_queue is a
        sequence of zero-based class indices. Runs restart their seeded RNG.
        """
        if self.inputs is None or self.rates is None:
            raise ValueError("sources and servers must be set")
        iterations = _int(iterations, "iterations")
        warmup = _int(warmup, "warmup", zero=True)
        if warmup >= iterations or not isinstance(saturated, bool) or not isinstance(record_trace, bool):
            raise ValueError("invalid warmup or bool flags")
        cap = None if admission_cap is None else _int(admission_cap, "admission_cap", zero=True)
        classes = len(self.inputs)
        if saturated and (arrivals is not None or not math.fsum(self.rates)):
            raise ValueError("saturated mode requires positive source weights and no arrival replay")
        if arrivals is not None:
            if len(arrivals) != iterations:
                raise ValueError("arrival replay length must equal iterations")
            for row in arrivals:
                if len(row) != classes:
                    raise ValueError("arrival replay must have one column per class")
                counts = [_int(x, "arrival count", zero=True) for x in row]
                if sum(counts) > 1_000_000:
                    raise ValueError("arrival replay batch is too large")
        rng = np.random.default_rng(self.seed)
        queue, active = deque(), []
        next_id = 0
        result = KvCacheCongestionSimResults()

        def make_job(k, age=0):
            nonlocal next_id
            job = {"id": next_id, "kind": k, "age": age}
            next_id += 1
            return job

        def kind_index(k):
            k = _int(k, "class index", zero=True)
            if k >= classes:
                raise ValueError("class index is out of range")
            return k

        def footprint(job):
            return self.inputs[job["kind"]] + 1 + job["age"]

        def profile():
            counts = [[0] * d for d in self.decodes]
            for job in active:
                counts[job["kind"]][job["age"]] += 1
            return counts

        if initial_state is not None:
            if len(initial_state) != classes:
                raise ValueError("initial_state requires one row per class")
            initial_counts = []
            for k, row in enumerate(initial_state):
                if len(row) != self.decodes[k]:
                    raise ValueError("initial_state shape mismatch")
                initial_counts.append([_int(x, "initial count", zero=True) for x in row])
            initial_memory = sum(
                (self.inputs[k] + 1 + j) * n for k, row in enumerate(initial_counts) for j, n in enumerate(row)
            )
            if initial_memory > self.memory_tokens:
                raise ValueError("initial_state exceeds memory")
            for k, row in enumerate(initial_counts):
                for age, count in enumerate(row):
                    active.extend(make_job(k, age) for _ in range(count))
        if initial_queue is not None:
            queue.extend(make_job(kind_index(k)) for k in initial_queue)
        result.initial_jobs = len(active) + len(queue)
        result.initial_work = sum(job["age"] for job in active)
        observation_completions = 0
        observation_memory = 0
        start = time.perf_counter()
        probabilities = np.array(self.rates) / math.fsum(self.rates) if saturated else None
        for step in range(iterations):
            result.executed_work += len(active)
            survivors = []
            completions = 0
            for job in active:
                job["age"] += 1
                if job["age"] == self.decodes[job["kind"]]:
                    completions += 1
                    result.completed_work += job["age"]
                else:
                    survivors.append(job)
            active = survivors
            result.completed += completions
            counts = (
                [0] * classes if saturated else (arrivals[step] if arrivals is not None else rng.poisson(self.rates))
            )
            if sum(counts) > 1_000_000:
                raise ValueError("Poisson batch is too large for request-level simulation")
            for k, count in enumerate(counts):
                queue.extend(make_job(k) for _ in range(int(count)))
                result.arrived += int(count)
            memory = sum(footprint(job) for job in active)
            post_execution_memory = memory
            evicted_ids = set()
            evicted = 0
            for job in sorted(active, key=lambda item: (item["age"], -item["id"])):
                if memory <= self.memory_tokens:
                    break
                memory -= footprint(job)
                evicted_ids.add(job["id"])
                result.discarded_work += job["age"]
                job["age"] = 0
                queue.append(job)
                evicted += 1
            active = [job for job in active if job["id"] not in evicted_ids]
            result.evictions += evicted
            admitted = 0
            while cap is None or admitted < cap:
                if not queue:
                    if not saturated:
                        break
                    queue.append(make_job(int(rng.choice(classes, p=probabilities))))
                    result.arrived += 1
                job = queue[0]
                if memory + footprint(job) > self.memory_tokens:
                    break
                active.append(queue.popleft())
                memory += footprint(job)
                admitted += 1
            if step >= warmup:
                observation_completions += completions
                observation_memory += memory
            if record_trace:
                result.iteration_trace.append(
                    {
                        "iteration": step,
                        "state": profile(),
                        "queue": len(queue),
                        "completed": completions,
                        "admitted": admitted,
                        "evicted": evicted,
                        "memory": memory,
                        "post_execution_memory": post_execution_memory,
                    }
                )
        result.unfinished = len(queue) + len(active)
        result.final_queue = len(queue)
        result.final_state = profile()
        result.remaining_work = sum(job["age"] for job in active)
        result.throughput = observation_completions / (iterations - warmup)
        result.memory_fraction = observation_memory / ((iterations - warmup) * self.memory_tokens)
        result.duration = time.perf_counter() - start
        return result
