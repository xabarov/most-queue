"""Gated ServerFilling extension with resource-holding checkpoint/resume costs.

The prefix rule is from Grosof and Harchol-Balter (2023),
doi:10.1145/3584684.3597264. Non-cancellable overhead phases and the gate on new
preemptions are explicit modelling choices here, NOT that paper's algorithm or
theorem. Useful work is preserved; no restart, I/O contention or memory model.
"""

import heapq
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob, RemainingPredictor, _integer, _Observation, _Replay
from most_queue.sim.utils.msj_packing import server_filling_selection
from most_queue.structs import MsjCheckpointResults, MsjSimulationResults


@dataclass(frozen=True)
class _Phase:
    need: int
    kind: str
    start: float
    end: float


@dataclass
class _CostReplay(_Replay):
    begun: set = field(default_factory=set)
    queue_wait: dict = field(default_factory=lambda: defaultdict(float))
    paused_wait: dict = field(default_factory=lambda: defaultdict(float))
    phase_times: dict = field(default_factory=lambda: {name: defaultdict(float) for name in ("checkpoint", "resume")})
    phase_segments: dict = field(default_factory=lambda: {name: [] for name in ("checkpoint", "resume")})
    phase_areas: dict = field(default_factory=lambda: defaultdict(float))
    protected_until: dict = field(default_factory=dict)
    review_at: float = float("inf")
    protected_preemptions: int = 0
    protection_expirations: int = 0


def _cost(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative duration")
    return float(value)


def _end(now, duration):
    end = now + duration
    if not np.isfinite(end) or end <= now:
        raise ValueError("phase duration overflows or is below timestamp precision; rescale the trace")
    return end


class MsjCheckpointSim(MsjGeneralSim):
    """Power-of-two MSJ packing with explicit, deterministic overhead times.

    Checkpoint holds the interrupted job's K servers without useful progress;
    after it finishes the job queues without servers. Resume holds K servers
    before useful service continues. First starts pay no resume cost. All useful
    progress is retained, even if a restored job is immediately preempted again.

    New preemption batches are gated while ANY checkpoint/resume is active.
    Existing useful jobs continue, and selected waiting jobs that actually fit
    can start. Overheads cannot be cancelled. The selector is the original SF
    prefix rule over all unfinished jobs, without S or predictions. This is a
    stated extension, not a Slurm emulation or a throughput-optimality claim.

    Optional min_service_time protects each new useful episode against early
    preemption. The clock resets AFTER resume, not at first arrival or during
    overhead. An expiry reconsiders the same prefix, without forcing a switch.
    Zero protection preserves EPIC-053; all three times zero delegate to the
    original SF. See docs/msj_checkpoint.md and docs/msj_protected_service.md.
    """

    # Preserve all four existing positional parameters; protection is opt-in.
    def __init__(  # pylint: disable=too-many-arguments
        self,
        k: int,
        checkpoint_time: float = 0.0,
        resume_time: float = 0.0,
        seed: int | None = None,
        *,
        min_service_time: float = 0.0,
    ) -> None:
        super().__init__(k, "server_filling", seed)
        self.checkpoint_time = _cost(checkpoint_time, "checkpoint_time")
        self.resume_time = _cost(resume_time, "resume_time")
        self.min_service_time = _cost(min_service_time, "min_service_time")

    def _service(self, trace, state, idx, now):
        if idx not in state.begun:
            state.starts[idx] = now
            state.begun.add(idx)
        end = _end(now, state.remaining.get(idx, trace[idx].service))
        state.completions[idx] = end
        state.active[idx] = _Phase(self.needs[trace[idx].cls], "service", now, end)
        if self.min_service_time:
            state.protected_until[idx] = _end(now, self.min_service_time)

    def _checkpoint(self, state, idx, now):
        running = state.active.pop(idx)
        state.remaining[idx] = running.end - now
        if now > running.start:
            state.segments.append((idx, running.start, now))
        state.preemptions[idx] = state.preemptions.get(idx, 0) + 1
        if self.checkpoint_time:
            state.active[idx] = _Phase(running.need, "checkpoint", now, _end(now, self.checkpoint_time))
        else:
            state.waiting.append(idx)
            state.queued_at[idx] = now

    def _admit(self, trace, state, idx, now):
        state.waiting.remove(idx)
        queued = now - state.queued_at.pop(idx, trace[idx].arrival)
        state.queue_wait[idx] += queued
        if idx in state.begun:
            state.paused_wait[idx] += queued
            if self.resume_time:
                state.active[idx] = _Phase(self.needs[trace[idx].cls], "resume", now, _end(now, self.resume_time))
                return
        self._service(trace, state, idx, now)

    def _dispatch_cost(self, trace, state, now):
        unfinished = heapq.merge(state.waiting, sorted(state.active))
        selected = server_filling_selection(self.k, ((idx, self.needs[trace[idx].cls]) for idx in unfinished))
        chosen = set(selected)
        if all(phase.kind == "service" for phase in state.active.values()):
            for idx in list(state.active):
                if idx not in chosen:
                    if state.protected_until.get(idx, now) > now:
                        state.protected_preemptions += 1
                    else:
                        self._checkpoint(state, idx, now)
        free = self.k - sum(phase.need for phase in state.active.values())
        for idx in selected:
            if idx not in state.active and self.needs[trace[idx].cls] <= free:
                self._admit(trace, state, idx, now)
                free -= self.needs[trace[idx].cls]
        state.waiting = deque(sorted(state.waiting))
        state.review_at = float("inf")
        if self.min_service_time and all(phase.kind == "service" for phase in state.active.values()):
            state.review_at = min(
                (
                    state.protected_until[idx]
                    for idx in state.active
                    if idx not in chosen and state.protected_until[idx] > now
                ),
                default=float("inf"),
            )

    def _finish_phases(self, trace, state, now):
        finished = [(idx, phase) for idx, phase in state.active.items() if phase.end <= now]
        for idx, phase in finished:
            del state.active[idx]
            if phase.kind == "service":
                state.segments.append((idx, phase.start, now))
            else:
                state.phase_times[phase.kind][idx] += now - phase.start
                state.phase_segments[phase.kind].append((idx, phase.start, now))
                if phase.kind == "checkpoint":
                    state.waiting.append(idx)
                    state.queued_at[idx] = now
                else:
                    self._service(trace, state, idx, now)
        state.waiting = deque(sorted(state.waiting))

    def _observe_cost(self, state, observation, now, nxt):
        dt = max(0.0, min(nxt, observation.end) - max(now, observation.begin))
        resources = defaultdict(int)
        for phase in state.active.values():
            resources[phase.kind] += phase.need
        allocated = sum(resources.values())
        for kind, count in resources.items():
            state.phase_areas[kind] += count * dt
        observation.busy_area += allocated * dt
        observation.idle_area += (self.k - allocated) * dt if state.waiting else 0.0
        if dt > 0:
            count = len(state.active) + len(state.waiting)
            observation.occupancy[count] = observation.occupancy.get(count, 0.0) + dt

    @staticmethod
    def _zero_result(result: MsjSimulationResults, trace, warmup):
        gaps, previous = defaultdict(float), {}
        for idx, start, end in sorted(result.service_segments):
            if idx in previous:
                gaps[idx] += start - previous[idx]
            previous[idx] = end
        samples = len(trace) - warmup
        return MsjCheckpointResults(
            **vars(result),
            first_wait_samples=[result.start_times[idx] - trace[idx].arrival for idx in range(warmup, len(trace))],
            interruption_samples=[gaps[idx] for idx in range(warmup, len(trace))],
            queue_wait_samples=result.wait_samples.copy(),
            checkpoint_samples=[0.0] * samples,
            resume_samples=[0.0] * samples,
            productive_utilization=result.utilization,
            checkpoint_utilization=0.0 if result.observation_time else None,
            resume_utilization=0.0 if result.observation_time else None,
        )

    def _cost_result(self, trace, warmup, state, observation):
        checkpoint, resume = state.phase_times["checkpoint"], state.phase_times["resume"]
        state.total_wait = {idx: state.queue_wait[idx] + checkpoint[idx] + resume[idx] for idx in range(len(trace))}
        result = self._trace_result(trace, warmup, state, observation)
        scale = self.k * result.observation_time
        return MsjCheckpointResults(
            **vars(result),
            first_wait_samples=[state.starts[idx] - trace[idx].arrival for idx in range(warmup, len(trace))],
            interruption_samples=[
                state.paused_wait[idx] + checkpoint[idx] + resume[idx] for idx in range(warmup, len(trace))
            ],
            queue_wait_samples=[state.queue_wait[idx] for idx in range(warmup, len(trace))],
            checkpoint_samples=[checkpoint[idx] for idx in range(warmup, len(trace))],
            resume_samples=[resume[idx] for idx in range(warmup, len(trace))],
            checkpoint_segments=state.phase_segments["checkpoint"],
            resume_segments=state.phase_segments["resume"],
            productive_utilization=state.phase_areas["service"] / scale if scale else None,
            checkpoint_utilization=state.phase_areas["checkpoint"] / scale if scale else None,
            resume_utilization=state.phase_areas["resume"] / scale if scale else None,
            checkpoint_resource_time=sum(self.needs[trace[idx].cls] * value for idx, value in checkpoint.items()),
            resume_resource_time=sum(self.needs[trace[idx].cls] * value for idx, value in resume.items()),
            protected_preemptions=state.protected_preemptions,
            protection_expirations=state.protection_expirations,
        )

    def run_trace(
        self,
        trace: Iterable[MsjTraceJob],
        warmup_jobs: int = 0,
        *,
        remaining_predictor: RemainingPredictor | None = None,
    ) -> MsjCheckpointResults:
        """Drain a common trace; phase endings precede arrivals and dispatch.

        Predictions are ignored, and remaining_predictor is rejected. Waiting
        includes queueing and both overheads; start_times are first USEFUL
        starts. Resource occupancy includes overhead while productive_utilization
        excludes it. No finite replay implies stability of the overhead model.
        """
        started = time.process_time()
        trace = tuple(trace)
        self._validate_trace(trace)
        warmup = _integer(warmup_jobs, "warmup_jobs", minimum=0)
        if warmup >= len(trace):
            raise ValueError("warmup_jobs must leave at least one measured job")
        if remaining_predictor is not None:
            raise ValueError("checkpoint packing does not support remaining_predictor")
        if self.checkpoint_time == self.resume_time == self.min_service_time == 0:
            return self._zero_result(super().run_trace(trace, warmup_jobs), trace, warmup)
        state = _CostReplay(np.zeros(len(trace)), np.zeros(len(trace)))
        observation = _Observation(trace[warmup].arrival, trace[-1].arrival, started)
        cursor, now = 0, 0.0
        while cursor < len(trace) or state.active or state.waiting:
            arrival = trace[cursor].arrival if cursor < len(trace) else float("inf")
            finish = min((phase.end for phase in state.active.values()), default=float("inf"))
            nxt = min(arrival, finish, state.review_at)
            if not np.isfinite(nxt):
                raise RuntimeError("checkpoint scheduler stalled with waiting jobs")
            self._observe_cost(state, observation, now, nxt)
            if nxt == state.review_at:
                state.protection_expirations += 1
            now = nxt
            self._finish_phases(trace, state, now)
            while cursor < len(trace) and trace[cursor].arrival <= now:
                state.waiting.append(cursor)
                cursor += 1
            self._dispatch_cost(trace, state, now)
        self.ttek = now
        return self._cost_result(trace, warmup, state, observation)
