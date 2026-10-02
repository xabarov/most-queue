"""General-service MSJ replay with backfilling and prediction-free packing.

EASY protects the predicted start of the first waiting job, not all jobs.
See Tsafrir, Etsion, Feitelson (2007), doi:10.1109/TPDS.2007.70606.
Conservative protects all waiting reservations; see Feitelson and Mu'alem Weil
(1998), doi:10.1109/IPPS.1998.669970, and sim.utils.msj_calendar.
Underestimated jobs are NOT killed: new backfills are suspended while any
running job is overdue. Reservations can consequently be violated. Actual
durations are used by the event engine, never by the reservation calculation.
"""

import heapq
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.sim.utils.msj_calendar import compress_reservations
from most_queue.sim.utils.msj_packing import PACKING_POLICIES, NonpreemptivePacking, server_filling_selection
from most_queue.structs import MsjSimulationResults

RemainingPredictor = Callable[[int, float], float | None]


@dataclass(frozen=True)
class MsjTraceJob:
    """One immutable input job; estimate is a duration, not an absolute deadline."""

    arrival: float
    cls: int
    service: float
    estimate: float | None = None


@dataclass(frozen=True)
class _Running:
    idx: int
    need: int
    estimate_end: float


@dataclass
class _Replay:
    starts: np.ndarray
    completions: np.ndarray
    waiting: deque = field(default_factory=deque)
    active: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    promises: dict = field(default_factory=dict)
    schedule: dict = field(default_factory=dict)
    runtime_updates: int = 0
    unavailable_runtime_updates: int = 0
    forecast_calendar_resets: int = 0
    packing: NonpreemptivePacking | None = None
    remaining: dict = field(default_factory=dict)
    episode_starts: dict = field(default_factory=dict)
    queued_at: dict = field(default_factory=dict)
    total_wait: dict = field(default_factory=dict)
    preemptions: dict = field(default_factory=dict)
    segments: list = field(default_factory=list)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class MsjGeneralSim(BaseSimulationCore):
    """Replay identical jobs under backfilling or prediction-free disciplines.

    ``set_servers(needs, distributions)`` takes library ``(params, notation)``
    pairs or callables ``sampler(rng)`` (e.g. lognormal/empirical sampling).
    For trace replay only, distributions may be omitted. ``run`` generates
    jobs before scheduling; ``run_trace`` drains every job, avoiding completion
    censoring. Neither a finite replay nor successful draining proves stability.
    MSFQ requires needs in {1, k}; ``msfq_threshold`` is an integer in [0, k).
    ServerFilling alone is preemptive-resume, with zero overhead and power-of-two
    k/needs. Its waiting time includes pauses. See docs/msj_packing.md.
    """

    def __init__(self, k: int, discipline: str = "fcfs", seed: int | None = None, *, msfq_threshold: int | None = None):
        super().__init__(seed=seed)
        self.k = _integer(k, "k")
        if discipline not in ("fcfs", "easy", "conservative", *PACKING_POLICIES):
            raise ValueError("unknown MSJ discipline")
        if msfq_threshold is not None and discipline != "msfq":
            raise ValueError("msfq_threshold is only valid for msfq")
        self.msfq_threshold = _integer(0 if msfq_threshold is None else msfq_threshold, "msfq_threshold", minimum=0)
        if self.msfq_threshold >= self.k:
            raise ValueError("msfq_threshold must be less than k")
        if discipline == "server_filling" and self.k & (self.k - 1):
            raise ValueError("server_filling requires power-of-two k")
        self.discipline = discipline
        self.needs = None
        self.rates = None
        self.samplers = None
        self.is_sources_set = self.is_servers_set = False

    def set_sources(self, rates):
        """Set one positive finite Poisson rate per class for generated traces."""
        rates = np.asarray(rates, dtype=float)
        if rates.ndim != 1 or not rates.size or not np.all(np.isfinite(rates)) or np.any(rates <= 0):
            raise ValueError("rates must be a nonempty vector of finite positive rates")
        self.rates = rates.copy()
        self.is_sources_set = True

    def set_servers(self, needs, distributions=None):
        """Set resource needs and optionally service generators for each class."""
        needs = [_integer(need, "server need") for need in needs]
        if not needs or max(needs) > self.k:
            raise ValueError("server needs must be nonempty, with 1 <= need <= k")
        if self.discipline == "msfq" and any(need not in (1, self.k) for need in needs):
            raise ValueError("msfq requires one-or-all needs in {1, k}")
        if self.discipline == "server_filling" and any(need & (need - 1) for need in needs):
            raise ValueError("server_filling requires power-of-two needs")
        samplers = None
        if distributions is not None:
            if len(distributions) != len(needs):
                raise ValueError("one service distribution is required per class")
            samplers = []
            for spec in distributions:
                if callable(spec):
                    samplers.append(lambda sampler=spec: sampler(self.generator))
                else:
                    params, notation = spec
                    if notation == "MAP":
                        raise ValueError("MAP is an arrival process, not an iid service distribution")
                    samplers.append(create_distribution(params, notation, self.generator).generate)
        self.needs, self.samplers = needs, samplers
        self.is_servers_set = True

    def make_trace(self, num_of_jobs: int, estimates=None) -> tuple[MsjTraceJob, ...]:
        """Generate a trace; estimates are None, explicit 'oracle', or class values.

        Service is sampled in arrival order before scheduling. No duration is
        inferred from an estimate. For noisy or feature-based predictions, make
        a trace and replace its estimates explicitly before replaying it.
        """
        count = _integer(num_of_jobs, "num_of_jobs")
        if not self.is_sources_set or self.samplers is None:
            raise ValueError("set_sources and set_servers with distributions are required")
        if len(self.rates) != len(self.needs):
            raise ValueError("arrival and service class counts must match")
        oracle = isinstance(estimates, str) and estimates == "oracle"
        if isinstance(estimates, str) and not oracle:
            raise ValueError("the only named estimate mode is 'oracle'")
        if estimates is not None and not oracle:
            estimates = np.asarray(estimates, dtype=float)
            if estimates.shape != self.rates.shape or not np.all(np.isfinite(estimates)) or np.any(estimates <= 0):
                raise ValueError("estimates must be one positive finite duration per class")
        total = self.rates.sum()
        arrivals = np.cumsum(self.generator.exponential(1 / total, size=count))
        classes = self.generator.choice(len(self.needs), size=count, p=self.rates / total)
        trace = []
        for arrival, cls in zip(arrivals, classes):
            service = float(self.samplers[cls]())
            estimate = service if oracle else (None if estimates is None else float(estimates[cls]))
            trace.append(MsjTraceJob(float(arrival), int(cls), service, estimate))
        self._validate_trace(trace, require_estimates=False)
        return tuple(trace)

    def run(
        self,
        num_of_jobs: int,
        warmup_fraction: float = 0.05,
        estimates=None,
        *,
        remaining_predictor: RemainingPredictor | None = None,
    ) -> MsjSimulationResults:
        """Measure num_of_jobs arrivals after a generated warm-up and drain them."""
        count = _integer(num_of_jobs, "num_of_jobs")
        if not np.isfinite(warmup_fraction) or not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        warmup = int(count * warmup_fraction)
        return self.run_trace(
            self.make_trace(count + warmup, estimates), warmup_jobs=warmup, remaining_predictor=remaining_predictor
        )

    def _validate_trace(self, trace, require_estimates=True):
        if not self.is_servers_set:
            raise ValueError("set_servers is required before replay")
        if not trace:
            raise ValueError("trace must not be empty")
        last = 0.0
        for job in trace:
            if not isinstance(job, MsjTraceJob):
                raise ValueError("trace entries must be MsjTraceJob objects")
            cls = _integer(job.cls, "job class", minimum=0)
            if cls >= len(self.needs):
                raise ValueError("job class is outside configured classes")
            if not np.isfinite(job.arrival) or job.arrival < last:
                raise ValueError("arrival times must be finite, nonnegative and nondecreasing")
            if not np.isfinite(job.service) or job.service <= 0:
                raise ValueError("service times must be finite and positive")
            if job.estimate is not None and (not np.isfinite(job.estimate) or job.estimate <= 0):
                raise ValueError("estimates must be finite and positive")
            if require_estimates and self.discipline in ("easy", "conservative") and job.estimate is None:
                raise ValueError("backfilling requires explicit estimates; select oracle explicitly if intended")
            last = job.arrival

    def _shadow(self, need, active, now):
        """Earliest predicted start using ONLY resource needs and forecast ends."""
        free = self.k - sum(job.need for job in active)
        if free >= need:
            return now
        for end, freed in sorted((job.estimate_end, job.need) for job in active):
            if end <= now:  # overdue: no finite release time is known
                continue
            free += freed
            if free >= need:
                return end
        return float("inf")

    def _dispatch(self, trace, state, now):
        waiting, active = state.waiting, state.active
        starts, completions = state.starts, state.completions
        events, promises = state.events, state.promises
        backfilled = violations = 0

        def start(idx):
            nonlocal violations
            job = trace[idx]
            starts[idx] = now
            completions[idx] = now + job.service  # event engine only
            estimate = None if self.discipline in PACKING_POLICIES else job.estimate
            estimate_end = float("inf") if estimate is None else now + estimate
            if not np.isfinite(completions[idx]) or (estimate is not None and not np.isfinite(estimate_end)):
                raise ValueError("completion/estimate times overflow; rescale the trace")
            if completions[idx] <= now or (estimate is not None and estimate_end <= now):
                raise ValueError("duration is below timestamp precision; rescale the trace")
            active[idx] = _Running(idx, self.needs[job.cls], estimate_end)
            heapq.heappush(events, (completions[idx], idx))
            if idx in promises and now > promises[idx] + 1e-10 * max(1.0, abs(promises[idx])):
                violations += 1

        if self.discipline == "server_filling":
            self._dispatch_server_filling(trace, state, now)
            return 0, 0
        if state.packing is not None:
            selected = state.packing.select(
                [(idx, self.needs[trace[idx].cls]) for idx in waiting],
                [(idx, job.need) for idx, job in active.items()],
            )
            for idx in selected:
                waiting.remove(idx)
                start(idx)
            return 0, 0
        if self.discipline == "conservative":
            backfilled = self._dispatch_conservative(trace, state, now, start)
            return backfilled, violations
        free = self.k - sum(job.need for job in active.values())
        while waiting and self.needs[trace[waiting[0]].cls] <= free:
            idx = waiting.popleft()
            start(idx)
            free -= self.needs[trace[idx].cls]
        if not waiting or self.discipline == "fcfs":
            return backfilled, violations
        head = waiting[0]
        need = self.needs[trace[head].cls]
        shadow = self._shadow(need, active.values(), now)
        if np.isfinite(shadow):
            promises[head] = min(promises.get(head, shadow), shadow)
        if not np.isfinite(shadow) or any(job.estimate_end <= now for job in active.values()):
            return backfilled, violations
        for idx in list(waiting)[1:]:
            job = trace[idx]
            if self.needs[job.cls] > free:
                continue
            candidate = _Running(idx, self.needs[job.cls], now + job.estimate)
            new_shadow = self._shadow(need, [*active.values(), candidate], now)
            if new_shadow <= shadow + 1e-12 * max(1.0, abs(shadow)):
                waiting.remove(idx)
                start(idx)
                free -= self.needs[job.cls]
                backfilled += 1
        return backfilled, violations

    def _dispatch_server_filling(self, trace, state, now):
        """Preempt/resume the chosen prefix packing; selection cannot see S."""
        unfinished = heapq.merge(state.waiting, sorted(state.active))
        selected = server_filling_selection(self.k, ((idx, self.needs[trace[idx].cls]) for idx in unfinished))
        chosen = set(selected)
        paused = []
        for idx in list(state.active):
            if idx not in chosen:
                state.remaining[idx] = state.completions[idx] - now
                state.segments.append((idx, state.episode_starts.pop(idx), now))
                state.queued_at[idx] = now
                state.preemptions[idx] = state.preemptions.get(idx, 0) + 1
                del state.active[idx]
                paused.append(idx)
        state.waiting.extend(paused)
        for idx in selected:
            if idx in state.active:
                continue
            state.waiting.remove(idx)
            if idx not in state.total_wait:
                state.starts[idx] = now
            state.total_wait[idx] = state.total_wait.get(idx, 0.0) + (
                now - state.queued_at.get(idx, trace[idx].arrival)
            )
            end = now + state.remaining.get(idx, trace[idx].service)
            if not np.isfinite(end) or end <= now:
                raise ValueError("completion overflows or duration is below timestamp precision; rescale the trace")
            state.completions[idx] = end
            state.episode_starts[idx] = now
            state.active[idx] = _Running(idx, self.needs[trace[idx].cls], float("inf"))
        if paused:
            state.waiting = deque(sorted(state.waiting))
        # Cancel stale completion events after preemption; at most k live jobs.
        state.events[:] = [(state.completions[idx], idx) for idx in state.active]
        heapq.heapify(state.events)

    def _dispatch_conservative(self, trace, state, now, start):
        """Keep every waiting reservation; use FCFS recovery on forecast overrun."""
        if any(job.estimate_end <= now for job in state.active.values()):
            # The forecast calendar is no longer feasible. Do not guess the
            # remaining durations or erase historical promises/missed starts.
            state.schedule.clear()
            free = self.k - sum(job.need for job in state.active.values())
            while state.waiting and self.needs[trace[state.waiting[0]].cls] <= free:
                idx = state.waiting.popleft()
                start(idx)
                free -= self.needs[trace[idx].cls]
            return 0
        requests = {idx: (self.needs[trace[idx].cls], trace[idx].estimate) for idx in state.waiting}
        state.schedule = compress_reservations(self.k, state.active.values(), requests, state.schedule, now)
        backfilled = 0
        for idx in sorted(state.schedule, key=lambda key: (state.schedule[key].start, key)):
            planned = state.schedule[idx].start
            if planned > now or idx in state.promises:
                state.promises[idx] = min(state.promises.get(idx, planned), planned)
            if planned == now:
                backfilled += idx != state.waiting[0]
                state.waiting.remove(idx)
                del state.schedule[idx]
                start(idx)
        return backfilled

    def _refresh_running(self, trace, state, now, predictor):
        """Refresh from class and elapsed service only; retain old promises."""
        updates = {}
        unavailable = 0
        invalidated = False
        for idx, running in state.active.items():
            residual = predictor(int(trace[idx].cls), float(now - state.starts[idx]))
            if residual is None:
                end = now  # no known release: reuse the overdue/FCFS recovery path
                unavailable += 1
            else:
                if (
                    isinstance(residual, (bool, np.bool_))
                    or not isinstance(residual, Real)
                    or not np.isfinite(residual)
                    or residual <= 0
                ):
                    raise ValueError("remaining_predictor must return a finite positive duration or None")
                end = float(now) + float(residual)
                if not np.isfinite(end):
                    raise ValueError("updated forecast overflows; rescale")
                if end <= now:
                    # At a discrete survival atom, subtracting a large start
                    # timestamp can leave a positive sub-ULP residual. There is
                    # no representable future release: freeze backfill, do not
                    # round up and invent a deadline or retry the same timer.
                    unavailable += 1
            invalidated |= end <= now or end > running.estimate_end
            updates[idx] = _Running(idx, running.need, end)
        if invalidated and state.schedule:
            state.schedule.clear()
            state.forecast_calendar_resets += 1
        state.active.update(updates)
        state.runtime_updates += len(updates)
        state.unavailable_runtime_updates += unavailable

    def run_trace(
        self, trace, warmup_jobs: int = 0, *, remaining_predictor: RemainingPredictor | None = None
    ) -> MsjSimulationResults:
        """Replay a sorted trace, preserving tie order; completions win time ties.

        All simultaneous arrivals are enqueued before scheduling. Time averages
        use [first measured arrival, last arrival]; they are None for a zero
        length interval. All measured jobs are drained for response statistics.

        Optional backfilling-only ``remaining_predictor(cls, age)`` is called
        for active jobs after observed completions and arrivals, before dispatch,
        at existing arrival/completion/reservation events. It cannot receive
        actual service or future completion from this interface. Initial waiting
        estimates remain explicit. None suspends new backfills; extended forecasts
        invalidate the conservative calendar but NEVER erase past promises.
        There is no heartbeat timer, online refit or hard coverage guarantee.
        A positive residual below timestamp resolution is also unavailable;
        it suspends backfill without inventing a later release timestamp.
        """
        started = time.process_time()
        trace = tuple(trace)
        self._validate_trace(trace)
        if remaining_predictor is not None and (
            not callable(remaining_predictor) or self.discipline not in ("easy", "conservative")
        ):
            raise ValueError("remaining_predictor must be callable and is only supported for backfilling")
        warmup = _integer(warmup_jobs, "warmup_jobs", minimum=0)
        if warmup >= len(trace):
            raise ValueError("warmup_jobs must leave at least one measured job")
        starts, completions = np.zeros(len(trace)), np.zeros(len(trace))
        state = _Replay(starts, completions)
        if self.discipline in PACKING_POLICIES and self.discipline != "server_filling":
            state.packing = NonpreemptivePacking(self.k, self.discipline, self.msfq_threshold)
        waiting, active, events, promises = state.waiting, state.active, state.events, state.promises
        begin, end = trace[warmup].arrival, trace[-1].arrival
        horizon = end - begin
        busy_area = idle_area = 0.0
        occupancy = defaultdict(float)
        backfilled = violations = 0
        cursor, now = 0, 0.0
        while cursor < len(trace) or events or waiting:
            next_arrival = trace[cursor].arrival if cursor < len(trace) else float("inf")
            next_done = events[0][0] if events else float("inf")
            next_reserved = min((slot.start for slot in state.schedule.values()), default=float("inf"))
            nxt = min(next_arrival, next_done, next_reserved)
            if not np.isfinite(nxt):
                raise RuntimeError("MSJ scheduler stalled with waiting jobs")
            dt = max(0.0, min(nxt, end) - max(now, begin))
            used = sum(job.need for job in active.values())
            busy_area += used * dt
            idle_area += (self.k - used) * dt if waiting else 0.0
            if dt > 0:
                occupancy[len(active) + len(waiting)] += dt
            now = nxt
            while events and events[0][0] <= now:
                _, idx = heapq.heappop(events)
                del active[idx]
                if self.discipline == "server_filling":
                    state.segments.append((idx, state.episode_starts.pop(idx), now))
            while cursor < len(trace) and trace[cursor].arrival <= now:
                waiting.append(cursor)
                cursor += 1
            if remaining_predictor is not None:
                self._refresh_running(trace, state, now, remaining_predictor)
            added, missed = self._dispatch(trace, state, now)
            waiting = state.waiting
            backfilled += added
            violations += missed
        self.ttek = now
        self.deadline_n = 0
        self.deadline_hits = dict.fromkeys(self.deadline_thresholds, 0)
        arrivals = np.array([job.arrival for job in trace])
        classes = np.array([job.cls for job in trace[warmup:]])
        waits = (starts - arrivals)[warmup:]
        if self.discipline == "server_filling":
            waits = np.array([state.total_wait[idx] for idx in range(warmup, len(trace))])
        sojourns = (completions - arrivals)[warmup:]
        for wait in waits:
            self._record_deadline_hit(float(wait))
        self.w = [float(np.mean(waits**order)) for order in range(1, 5)]
        self.v = [float(np.mean(sojourns**order)) for order in range(1, 5)]
        quantiles = (0.95, 0.99)

        def quantile_map(values):
            return {q: float(np.quantile(values, q)) if values.size else float("nan") for q in quantiles}

        w_class, v_class, counts, tails = [], [], [], []
        for cls in range(len(self.needs)):
            mask = classes == cls
            counts.append(int(mask.sum()))
            w_class.append(float(waits[mask].mean()) if mask.any() else float("nan"))
            v_class.append(float(sojourns[mask].mean()) if mask.any() else float("nan"))
            tails.append(quantile_map(sojourns[mask]))
        p = None
        if horizon > 0:
            p = [occupancy[i] / horizon for i in range(max(occupancy) + 1)]
        return MsjSimulationResults(
            w=self.w,
            v=self.v,
            p=p,
            w_per_class=w_class,
            v_per_class=v_class,
            utilization=busy_area / (self.k * horizon) if horizon else None,
            idle_with_queue=idle_area / (self.k * horizon) if horizon else None,
            throughput=float(np.sum((completions > begin) & (completions <= end)) / horizon) if horizon else None,
            duration=time.process_time() - started,
            start_times=starts.tolist(),
            completion_times=completions.tolist(),
            wait_samples=waits.tolist(),
            sojourn_samples=sojourns.tolist(),
            counts_per_class=counts,
            w_quantiles=quantile_map(waits),
            v_quantiles=quantile_map(sojourns),
            v_quantiles_per_class=tails,
            backfilled=backfilled,
            reservations=len(promises),
            reservation_violations=violations,
            reserved_start_times=dict(promises),
            observation_time=horizon,
            runtime_updates=state.runtime_updates,
            unavailable_runtime_updates=state.unavailable_runtime_updates,
            forecast_calendar_resets=state.forecast_calendar_resets,
            preemptions=sum(state.preemptions.values()),
            preemptions_per_job=(
                [state.preemptions.get(idx, 0) for idx in range(len(trace))]
                if self.discipline == "server_filling"
                else []
            ),
            service_segments=state.segments,
        )
