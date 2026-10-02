"""Opt-in nonpreemptive MSJ carry-in and labelled resource-release replay.

Uses unchanged MsjGeneralSim dispatchers. A cancelled job supplies occupied time
to cancellation, NOT latent completion demand; no queue-abandonment clock is
inferred. Runtime limits apply after start and never convert timeout to success.
"""

import heapq
from dataclasses import dataclass
from numbers import Real

import numpy as np

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob, _Replay, _Running
from most_queue.sim.utils.msj_packing import NonpreemptivePacking

LIFECYCLE_POLICIES = ("fcfs", "first_fit", "msf", "adaptive_quickswap", "easy", "conservative")


@dataclass(frozen=True)
class MsjLifecycleJob(MsjTraceJob):
    """Positive service-to-outcome; an optional budget limits occupied runtime.

    outcome is completed/cancelled; service > runtime_limit becomes timed_out.
    Equality preserves the original outcome. Neither label nor budget is sent
    to the dispatcher; the supplied estimate stays unchanged.
    """

    outcome: str = "completed"
    runtime_limit: float | None = None


@dataclass(frozen=True)
class MsjCarryIn:
    """An already running job at time zero with elapsed service age.

    job.arrival must be zero; full service includes age. No retroactive budget
    may be specified. Unknown/expired forecast gives an overdue running job.
    """

    job: MsjLifecycleJob
    age: float


@dataclass
class MsjLifecycleResults:
    """Release arrays cover running, waiting, then new arrivals in that order.

    Initial running starts are negative ages; consumed_services count only work
    after time zero. Outcomes are not interchangeable with successful completion.
    Time averages exclude drain; resource_time_by_outcome includes all drain.
    """

    start_times: list[float]
    release_times: list[float]
    outcomes: list[str]
    consumed_services: list[float]
    initial_count: int
    utilization: float | None
    idle_with_queue: float | None
    observation_time: float
    resource_time_by_outcome: dict[str, float]
    backfilled: int
    reservation_violations: int
    reserved_start_times: dict[int, float]


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


class MsjLifecycleSim(MsjGeneralSim):
    """Separate run_lifecycle API; inherited ordinary replay is unchanged.

    Initial queue order is the supplied order, ahead of arrivals at time zero.
    Existing reservations/Adaptive phase are reset. Only six nonpreemptive
    policies are supported; there is no pause/resume or unknown waiting cancel.
    """

    def __init__(self, k, discipline="fcfs", seed=None):
        if discipline not in LIFECYCLE_POLICIES:
            raise ValueError("lifecycle replay requires a supported nonpreemptive policy")
        super().__init__(k, discipline, seed)

    def run_lifecycle(self, trace, *, initial_running=(), initial_waiting=(), observation_window=None):
        """Replay to resource release, seeding active jobs before any dispatch."""
        trace, running, waiting = tuple(trace), tuple(initial_running), tuple(initial_waiting)
        if not trace:
            raise ValueError("at least one new arrival is required")
        for item in running:
            if not isinstance(item, MsjCarryIn):
                raise ValueError("initial_running entries must be MsjCarryIn")
        original = tuple(item.job for item in running) + waiting + trace
        for job in original:
            if not isinstance(job, MsjLifecycleJob) or job.outcome not in ("completed", "cancelled"):
                raise ValueError("entries must be lifecycle jobs with completed/cancelled outcome")
            if job.runtime_limit is not None and _nonnegative(job.runtime_limit, "runtime_limit") == 0:
                raise ValueError("runtime_limit must be positive when supplied")
        self._validate_trace(original, require_estimates=False)
        self._validate_trace(waiting + trace)
        initial_count = len(running) + len(waiting)
        if any(job.arrival != 0 or job.runtime_limit is not None for job in original[:initial_count]):
            raise ValueError("initial jobs must have arrival=0 and no retroactive runtime_limit")
        if sum(self.needs[item.job.cls] for item in running) > self.k:
            raise ValueError("initial running jobs exceed capacity")
        ages = [_nonnegative(item.age, "age") for item in running]
        if any(age >= item.job.service for age, item in zip(ages, running)):
            raise ValueError("initial service age must leave positive remaining service")
        used, outcomes, effective = [], [], []
        for idx, job in enumerate(original):
            age = ages[idx] if idx < len(running) else 0.0
            service = min(job.service, job.runtime_limit) if job.runtime_limit is not None else job.service
            outcomes.append("timed_out" if service < job.service else job.outcome)
            used.append(float(service - age))
            effective.append(MsjTraceJob(job.arrival, job.cls, float(service - age), job.estimate))
        if observation_window is None:
            begin, end = trace[0].arrival, trace[-1].arrival
        else:
            if len(observation_window) != 2:
                raise ValueError("observation_window must be a pair")
            begin, end = (_nonnegative(value, "observation bound") for value in observation_window)
        if not 0 <= begin <= end <= trace[-1].arrival:
            raise ValueError("observation window must end no later than the last arrival")
        state = _Replay(np.zeros(len(original)), np.zeros(len(original)))
        if self.discipline in ("first_fit", "msf", "adaptive_quickswap"):
            state.packing = NonpreemptivePacking(self.k, self.discipline)
        for idx, item in enumerate(running):
            estimate = item.job.estimate
            estimate_end = max(0.0, estimate - ages[idx]) if estimate is not None else 0.0
            state.starts[idx], state.completions[idx] = -ages[idx], used[idx]
            state.active[idx] = _Running(idx, self.needs[item.job.cls], estimate_end)
            heapq.heappush(state.events, (used[idx], idx))
        state.waiting.extend(range(len(running), initial_count))
        cursor, now = initial_count, 0.0
        busy_area = idle_area = 0.0
        backfilled = violations = 0
        first = True
        while first or cursor < len(original) or state.events or state.waiting:
            candidates = [
                effective[cursor].arrival if cursor < len(original) else float("inf"),
                state.events[0][0] if state.events else float("inf"),
                min((slot.start for slot in state.schedule.values()), default=float("inf")),
            ]
            nxt = 0.0 if first else min(candidates)
            first = False
            if not np.isfinite(nxt) or nxt < now:
                raise RuntimeError("lifecycle scheduler stalled or moved backwards")
            dt = max(0.0, min(nxt, end) - max(now, begin))
            occupied = sum(job.need for job in state.active.values())
            busy_area += occupied * dt
            idle_area += (self.k - occupied) * dt if state.waiting else 0.0
            now = nxt
            while state.events and state.events[0][0] <= now:
                _, idx = heapq.heappop(state.events)
                del state.active[idx]
            while cursor < len(original) and effective[cursor].arrival <= now:
                state.waiting.append(cursor)
                cursor += 1
            added, missed = self._dispatch(effective, state, now)
            backfilled += added
            violations += missed
        self.ttek = now
        horizon = end - begin
        work = dict.fromkeys(("completed", "cancelled", "timed_out"), 0.0)
        for job, duration, outcome in zip(original, used, outcomes):
            work[outcome] += self.needs[job.cls] * duration
        return MsjLifecycleResults(
            state.starts.tolist(),
            state.completions.tolist(),
            outcomes,
            used,
            initial_count,
            busy_area / (self.k * horizon) if horizon else None,
            idle_area / (self.k * horizon) if horizon else None,
            horizon,
            work,
            backfilled,
            violations,
            dict(state.promises),
        )
