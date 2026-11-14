"""Input polling and a bounded queue of planned iterations on one serial GPU.

The specified DES follows the control flow, not individual kernels, of vLLM
v0.32.0 EngineCore.step_with_batch_queue and AsyncScheduler placeholders:
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/engine/core.py
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/async_scheduler.py

CPU scheduling/polling is instantaneous. An iteration's GPU work is serial;
up to two iterations can be planned before the oldest result is consumed.
These assumptions expose admission delay without a fitted busy-state offset.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from most_queue.sim.advance_reservation import AdvanceReservationSim
from most_queue.structs import PrefillDecodeSimResults


@dataclass
class IterationAdmissionResults(PrefillDecodeSimResults):
    """Ready-arrival statistics with input polling separated from slot waiting.

    w includes both input waiting and waiting for an active slot. start_times
    marks slot reservation; engine_entry_times marks the preceding input poll.
    Population counts all ready arrivals, including the unpolled input queue.
    Occupancy excludes the final drain, as in PrefillDecodeSimResults.
    """

    engine_entry_times: list[float] = field(default_factory=list)
    input_wait_times: list[float] = field(default_factory=list)
    input_wait_mean: float = 0.0


class IterationAdmissionSim(AdvanceReservationSim):
    """FCFS slot admission after input polling, with one or two planned batches.

    Arrivals are requests ready for EngineCore input, before QUEUED. Each poll
    drains inputs, then one batch is scheduled. A nonfull batch queue permits
    another immediate poll/schedule; a full queue waits for its oldest result.
    Future actions are immutable. Planning advances output placeholders, while
    slots are released only when the terminal result actually completes.

    poll_inputs=False is an ablation: inputs register at arrival even while the
    core is waiting. It changes the division of input/slot waiting, not the GPU
    timeline. This finite, unchunked, no-preemption DES is not a vLLM emulator.
    """

    # Same explicit stage-cost and finite-cohort API as the original simulator.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        batch_cap=256,
        token_budget=8192,
        seed=None,
        max_iterations=2_000_000,
        *,
        inflight_batches=2,
        poll_inputs=True,
    ):
        super().__init__(batch_cap, token_budget, seed, max_iterations)
        if (
            isinstance(inflight_batches, bool)
            or not isinstance(inflight_batches, int)
            or inflight_batches not in (1, 2)
        ):
            raise ValueError("inflight_batches must be 1 or 2")
        if not isinstance(poll_inputs, bool):
            raise ValueError("poll_inputs must be Boolean")
        self.inflight_batches, self.poll_inputs = inflight_batches, poll_inputs

    @staticmethod
    def _vector(values, name):
        try:
            objects = np.asarray(values, dtype=object)
            if objects.ndim != 1 or any(isinstance(v, (bool, np.bool_)) for v in objects):
                raise ValueError("Boolean or multidimensional vector")
            result = np.asarray(values, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a finite numeric vector") from error
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite numeric vector")
        return result

    def _lengths(self, inputs, outputs):
        inputs = self._vector(inputs, "inputs")
        outputs = self._vector(outputs, "outputs")
        super()._lengths(inputs, outputs)

    def replay(self, arrivals, inputs, outputs, warmup_fraction=0.0, *, record_iterations=False):
        """Drain an ordered ready-arrival cohort and return all request clocks.

        At a tie, existing GPU results complete before slot admission; all tied
        ready arrivals join the same poll. Statistics retain full timestamp
        arrays and remove warmup only from aggregate latency/occupancy measures.
        The explicit iteration budget counts nonempty planned GPU batches.
        """
        if self.alpha is None:
            raise ValueError("set_servers must be called before replay")
        arrivals = self._vector(arrivals, "arrivals")
        if arrivals.size == 0 or arrivals[0] < 0 or np.any(np.diff(arrivals) < 0):
            raise ValueError("arrivals must be nonempty, nonnegative and nondecreasing")
        self._lengths(inputs, outputs)
        if len(inputs) != len(arrivals) or len(outputs) != len(arrivals):
            raise ValueError("arrival and length vectors must have the same size")
        if isinstance(warmup_fraction, (bool, np.bool_)) or not np.isfinite(warmup_fraction):
            raise ValueError("warmup_fraction must be in [0,1)")
        if not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0,1)")
        started = time.process_time()
        inputs, outputs = np.asarray(inputs, dtype=int), np.asarray(outputs, dtype=int)
        count = len(arrivals)
        entries, starts, computes, prefills, completes = (np.full(count, np.nan) for _ in range(5))
        planned = np.zeros(count, dtype=int)
        tokens = [[] for _ in range(count)]
        incoming, waiting, batches = deque(), deque(), deque()
        active, trace = [], []
        cursor = departed = iterations = 0
        now = 0.0
        warmup = int(count * warmup_fraction)
        begin, end = arrivals[warmup], arrivals[-1]
        state_area = {}
        active_area = busy_area = 0.0

        def integrate(target):
            nonlocal now, active_area, busy_area
            dt = max(0.0, min(target, end) - max(now, begin))
            if dt:
                population = cursor - departed
                state_area[population] = state_area.get(population, 0.0) + dt
                active_area += dt * len(active)
                busy_area += dt * bool(batches)
            now = target

        def advance(target):
            nonlocal cursor
            while cursor < count and arrivals[cursor] <= target:
                integrate(float(arrivals[cursor]))
                if self.poll_inputs:
                    incoming.append(cursor)
                else:
                    entries[cursor] = now
                    waiting.append(cursor)
                cursor += 1
            integrate(target)

        def poll():
            while incoming:
                job = incoming.popleft()
                entries[job] = now
                waiting.append(job)

        def plan():
            nonlocal iterations
            while waiting and len(active) < self.batch_cap:
                job = waiting.popleft()
                starts[job] = now
                active.append(job)
            actions, costs = [], [self.alpha]
            for job in active:
                if planned[job] == outputs[job]:
                    continue
                prefill = planned[job] == 0
                amount = int(inputs[job]) if prefill else 1
                context = inputs[job] if prefill else inputs[job] + planned[job]
                costs.append(self.beta * amount + self.gamma * context)
                actions.append((job, "prefill" if prefill else "decode", amount))
            if not actions:
                return False
            try:
                work = math.fsum(costs)
            except OverflowError as error:
                raise ValueError("iteration duration overflowed") from error
            start = max(now, batches[-1]["end"]) if batches else now
            finish = start + work
            if not math.isfinite(finish) or finish <= start:
                raise ValueError("iteration duration overflowed or failed to advance the clock")
            iterations += 1
            if iterations > self.max_iterations:
                raise RuntimeError("max_iterations exhausted; increase the explicit simulation budget")
            for job, _, _ in actions:
                planned[job] += 1
                if np.isnan(computes[job]):
                    computes[job] = start
            batch = {
                "scheduled": now,
                "start": start,
                "end": finish,
                "actions": actions,
                "active": active.copy(),
                "inflight_after_submit": len(batches) + 1,
            }
            batches.append(batch)
            if record_iterations:
                trace.append(batch)
            return True

        while cursor < count or incoming or waiting or active or batches:
            if not (incoming or waiting or active or batches):
                advance(float(arrivals[cursor]))
            poll()
            submitted = plan()
            if submitted and len(batches) < self.inflight_batches:
                continue
            if not batches:
                raise RuntimeError("no schedulable actions or pending GPU results")
            batch = batches[0]
            advance(batch["end"])
            batches.popleft()
            finished = set()
            for job, stage, _ in batch["actions"]:
                if stage == "prefill":
                    prefills[job] = now
                tokens[job].append(now)
                if len(tokens[job]) == outputs[job]:
                    completes[job] = now
                    finished.add(job)
            active = [job for job in active if job not in finished]
            departed += len(finished)

        selected = slice(warmup, None)
        waits = starts[selected] - arrivals[selected]
        sojourns = completes[selected] - arrivals[selected]
        first = np.asarray([row[0] for row in tokens])
        input_waits = entries - arrivals
        gaps = [b - a for row in tokens[warmup:] for a, b in zip(row, row[1:])]
        observation = end - begin
        pmf = [state_area.get(i, 0.0) / observation for i in range(max(state_area) + 1)] if observation else None
        jobs_mean = sum(i * area for i, area in state_area.items()) / observation if observation else 0.0
        batch_mean = active_area / observation if observation else 0.0
        busy = busy_area / observation if observation else 0.0
        return IterationAdmissionResults(
            w=[float(np.mean(waits**r)) for r in range(1, 4)],
            v=[float(np.mean(sojourns**r)) for r in range(1, 4)],
            p=pmf,
            utilization=busy,
            ttft_mean=float(np.mean(first[selected] - arrivals[selected])),
            itl_mean=float(np.mean(gaps)) if gaps else None,
            prefill_mean=float(np.mean(prefills[selected] - starts[selected])),
            first_decode_mean=0.0,
            all_decode_mean=float(np.sum(completes[selected] - prefills[selected]) / np.sum(outputs[selected])),
            service_mean=float(np.mean(completes[selected] - starts[selected])),
            batch_mean=batch_mean,
            jobs_mean=jobs_mean,
            queue_mean=jobs_mean - batch_mean,
            busy_probability=busy,
            waiting_probability=float(np.mean(waits > 0)),
            observation_time=float(observation),
            start_times=starts.tolist(),
            first_compute_times=computes.tolist(),
            prefill_end_times=prefills.tolist(),
            completion_times=completes.tolist(),
            output_times=tokens,
            iteration_trace=trace,
            engine_entry_times=entries.tolist(),
            input_wait_times=input_waits.tolist(),
            input_wait_mean=float(np.mean(input_waits[selected])),
            duration=time.process_time() - started,
        )
