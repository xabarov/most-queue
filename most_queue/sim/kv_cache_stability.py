"""Independent token-stage simulator for KV-constrained constant-duration slots.

Nie, Si and Zhou, arXiv:2605.04595v1, Sections 3--4 / Appendix A.
An explicit realization of maximal packing with zero-cost CPU offload: only
selected requests occupy GPU KV memory; all unselected partial requests can
be swapped out for free. This is NOT a hard-resident vLLM simulator.
No analytical calculator imports, exponential service, or final draining.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from most_queue.structs import KvCacheSimResults


@dataclass
class _Request:
    """One unfinished request; stage counts progress only when selected."""

    identity: int
    lengths: tuple[int, int]
    stage: int = 0


class KvCacheStabilitySim:
    """Finite-horizon simulator: ongoing first, then scanning the waiting FIFO.

    Skipping non-fitting ongoing requests is mandatory for maximality; strict
    head-of-line blocking is a different policy. Each request executes at most
    one stage per slot. Arrivals occur at slot beginnings and can join that slot.
    Output length means the number of charged decode stages after prefill.
    """

    def __init__(self, memory_tokens, chunk_size=1, seed=None):
        for name, value in [("memory_tokens", memory_tokens), ("chunk_size", chunk_size)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.memory_tokens = int(memory_tokens)
        self.chunk_size = int(chunk_size)
        self.seed = seed
        self.l = self.slot_time = None
        self._lengths = self._probabilities = None
        self._stages = {}

    def set_sources(self, l):
        """Set Poisson rate per time unit; replay bypasses arrival generation."""
        if isinstance(l, bool) or not isinstance(l, Real) or not np.isfinite(l) or l < 0:
            raise ValueError("arrival rate must be finite and nonnegative")
        self.l = float(l)

    def set_servers(self, joint_pmf, slot_time):
        """Set a finite joint PMF and positive constant time per slot.

        Stage footprints are independently enumerated, not a closed-form g.
        Each supported type is limited to one million stages for safe replay.
        """
        if (
            isinstance(slot_time, bool)
            or not isinstance(slot_time, Real)
            or not np.isfinite(slot_time)
            or slot_time <= 0
        ):
            raise ValueError("slot_time must be positive and finite")
        if not isinstance(joint_pmf, Mapping) or not joint_pmf:
            raise ValueError("joint_pmf must be a nonempty mapping")
        stages, probabilities = {}, []
        for lengths, p in joint_pmf.items():
            if not isinstance(lengths, tuple) or len(lengths) != 2:
                raise ValueError("joint PMF keys must be length pairs")
            if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in lengths):
                raise ValueError("lengths must be positive integers")
            s, o = map(int, lengths)
            if s % self.chunk_size:
                raise ValueError("input_tokens must be a multiple of chunk_size")
            if isinstance(p, bool) or not isinstance(p, Real) or not np.isfinite(p) or p < 0:
                raise ValueError("probabilities must be finite and nonnegative")
            if p == 0:
                continue
            if s + o > self.memory_tokens:
                raise ValueError("peak footprint exceeds memory_tokens")
            if s // self.chunk_size + o > 1_000_000:
                raise ValueError("request type exceeds the one-million-stage replay limit")
            stages[s, o] = tuple(range(self.chunk_size, s + 1, self.chunk_size)) + tuple(range(s + 1, s + o + 1))
            probabilities.append(float(p))
        total = sum(probabilities)
        if not np.isclose(total, 1, rtol=0, atol=1e-10):
            raise ValueError("joint probabilities must sum to one")
        self._stages = stages
        self._lengths = tuple(stages)
        self._probabilities = np.array(probabilities) / total
        self.slot_time = float(slot_time)

    def run(self, num_of_slots, *, warmup_slots=0, arrivals_by_slot=None, record_trace=False):
        """Run exactly num_of_slots; unfinished jobs remain at the horizon.

        arrivals_by_slot is a sequence of lists of supported (s,o) pairs,
        one list per slot. All counters include warmup; throughput and resource
        fractions use only slots >= warmup_slots. Repeated seeded runs replay.
        Trace includes before/after remaining work and exact next-stage charges.
        No moments of completed-only latencies are reported as queue means.
        """
        if self.l is None or self.slot_time is None:
            raise ValueError("set_sources and set_servers must be called first")
        if (
            isinstance(num_of_slots, bool)
            or not isinstance(num_of_slots, Integral)
            or not 1 <= num_of_slots <= 2_000_000
        ):
            raise ValueError("num_of_slots must be an integer in 1..2000000")
        if (
            isinstance(warmup_slots, bool)
            or not isinstance(warmup_slots, Integral)
            or not 0 <= warmup_slots < num_of_slots
        ):
            raise ValueError("warmup_slots must be an integer in 0..num_of_slots-1")
        if not isinstance(record_trace, bool):
            raise ValueError("record_trace must be bool")
        if arrivals_by_slot is not None and len(arrivals_by_slot) != num_of_slots:
            raise ValueError("arrivals_by_slot must have one entry per slot")
        if not np.isfinite(num_of_slots * self.slot_time) or not np.isfinite(self.l * self.slot_time):
            raise ValueError("time or arrival arithmetic overflowed")
        started = time.process_time()
        rng = np.random.default_rng(self.seed)
        active, waiting, trace = [], [], []
        arrived = completed = input_area = fulfilled = outstanding = swapped = 0
        observed_completions = observed_area = busy_slots = 0
        for slot in range(num_of_slots):
            if arrivals_by_slot is None:
                count = rng.poisson(self.l * self.slot_time)
                if arrived + count > 2_000_000:
                    raise ValueError("simulation exceeds the two-million-request safety limit")
                indices = rng.choice(len(self._lengths), size=count, p=self._probabilities)
                arrivals = [self._lengths[index] for index in indices]
            else:
                arrivals = arrivals_by_slot[slot]
            incoming = 0
            if arrived + len(arrivals) > 2_000_000:
                raise ValueError("simulation exceeds the two-million-request safety limit")
            for lengths in arrivals:
                if (
                    not isinstance(lengths, tuple)
                    or len(lengths) != 2
                    or any(isinstance(x, bool) or not isinstance(x, Integral) for x in lengths)
                    or lengths not in self._stages
                ):
                    raise ValueError("replayed length pair must belong to positive-probability support")
                waiting.append(_Request(arrived, lengths))
                arrived += 1
                incoming += sum(self._stages[lengths])
            input_area += incoming
            outstanding += incoming
            before = outstanding
            backlog_before = len(active) + len(waiting)
            available = self.memory_tokens
            new_active, new_waiting, selected = [], [], []
            old_swapped = 0
            # Unselected partial requests reside on CPU; there is no hidden GPU
            # reservation that would consume capacity without fulfilling work.
            for pool, was_active in [(active, True), (waiting, False)]:
                for request in pool:
                    charge = self._stages[request.lengths][request.stage]
                    if charge > available:
                        if was_active:
                            new_active.append(request)
                            old_swapped += 1
                        else:
                            new_waiting.append(request)
                        continue
                    available -= charge
                    if record_trace:
                        selected.append((request.identity, request.stage, charge))
                    request.stage += 1
                    if request.stage == len(self._stages[request.lengths]):
                        completed += 1
                        observed_completions += slot >= warmup_slots
                    else:
                        new_active.append(request)
            active, waiting = new_active, new_waiting
            served = self.memory_tokens - available
            fulfilled += served
            outstanding -= served
            swapped += old_swapped
            if slot >= warmup_slots:
                observed_area += served
                busy_slots += served > 0
            if record_trace:
                trace.append(
                    {
                        "slot": slot,
                        "backlog_before": backlog_before,
                        "backlog_after": len(active) + len(waiting),
                        "incoming_area": incoming,
                        "work_before": before,
                        "fulfilled": served,
                        "work_after": outstanding,
                        "selected": selected,
                        "swapped": old_swapped,
                    }
                )
        observed_slots = num_of_slots - warmup_slots
        return KvCacheSimResults(
            arrived=arrived,
            completed=completed,
            unfinished=len(active) + len(waiting),
            input_memory_area=input_area,
            fulfilled_memory_area=fulfilled,
            outstanding_memory_area=outstanding,
            throughput=observed_completions / (observed_slots * self.slot_time),
            memory_fraction=observed_area / (observed_slots * self.memory_tokens),
            utilization=busy_slots / observed_slots,
            swapped_request_slots=swapped,
            observation_time=observed_slots * self.slot_time,
            slot_trace=trace,
            duration=time.process_time() - started,
        )
