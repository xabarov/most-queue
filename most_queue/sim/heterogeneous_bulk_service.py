"""Independent request-level DES for heterogeneous exponential bulk service.

No analytical kernel or geometric/PH decomposition is imported. Fixed horizon,
no drain. Admission-cohort tails have sufficient follow-up for every requested
time, including unfinished jobs; completed-only latency moments are not reported.
"""

import heapq
import math
import numbers
from collections import deque

import numpy as np

from most_queue.structs import HeterogeneousBulkSimResults
from most_queue.theory.base_queue import BaseQueue


class BulkServiceHeterogeneousSim(BaseQueue):
    """Shared FCFS queue with individual servers; threshold then input priority."""

    def __init__(self, counts, seed=None):
        counts = tuple(counts)
        if not counts or any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1 for n in counts):
            raise ValueError("counts must be positive integers")
        if sum(counts) > 20000:
            raise ValueError("simulation supports at most 20000 servers")
        super().__init__(int(sum(counts)))
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        self.counts = tuple(int(n) for n in counts)
        self.seed = seed
        self.l = None
        self.rates = self.thresholds = self.capacities = None

    @staticmethod
    def _real(value, name, positive=True):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (positive and value == 0):
            raise ValueError(f"{name} is outside its valid range")
        return float(value)

    def set_sources(self, l):
        """Set positive Poisson rate."""
        self.l = self._real(l, "arrival rate")
        self.is_sources_set = True

    def set_servers(self, rates, thresholds, capacities):
        """Set exponential batch rates, minimum and maximum sizes by input type."""
        rates = tuple(self._real(x, "service rate") for x in rates)
        thresholds, capacities = tuple(thresholds), tuple(capacities)
        if not len(rates) == len(thresholds) == len(capacities) == len(self.counts):
            raise ValueError("type array lengths must match counts")
        if any(isinstance(x, bool) or not isinstance(x, numbers.Integral) for x in thresholds + capacities):
            raise ValueError("thresholds and capacities must be integers")
        if any(not 1 <= t <= b <= 1024 for t, b in zip(thresholds, capacities)):
            raise ValueError("require 1 <= threshold <= capacity <= 1024")
        self.rates = rates
        self.thresholds, self.capacities = thresholds, capacities
        self.is_servers_set = True

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        horizon,
        warmup=0,
        tail_times=(),
        cohort_end=None,
        arrivals=None,
        service_times=None,
        record_trace=False,
        max_events=2000000,
    ):
        """Run from empty state; replay arrivals and per-type durations optionally.

        Events exactly at horizon are excluded; completions precede tied arrivals.
        Tails use arrivals in [warmup, cohort_end), with cohort_end <= horizon-max(t).
        Replay service_times is one finite positive-duration sequence per type.
        Exhaustion raises instead of silently reverting to random service.
        """
        self._check_if_servers_and_sources_set()
        start = self._measure_time()
        horizon = self._real(horizon, "horizon")
        warmup = self._real(warmup, "warmup", False)
        if warmup >= horizon:
            raise ValueError("warmup must be smaller than horizon")
        tails = [self._real(t, "tail time", False) for t in tail_times]
        cohort_end = (warmup + horizon) / 2 if cohort_end is None else self._real(cohort_end, "cohort end")
        if not warmup < cohort_end <= horizon - max(tails, default=0):
            raise ValueError("cohort needs positive width and full follow-up for tail_times")
        if isinstance(max_events, bool) or not isinstance(max_events, numbers.Integral) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        replay = None
        if arrivals is not None:
            replay = [self._real(t, "arrival time", False) for t in arrivals]
            if any(t >= horizon for t in replay) or any(b < a for a, b in zip(replay, replay[1:])):
                raise ValueError("replay arrivals must be sorted in [0,horizon)")
        durations = None
        if service_times is not None:
            profiles = [list(self._real(x, "service duration") for x in profile) for profile in service_times]
            if len(profiles) != len(self.counts):
                raise ValueError("one duration replay per server type is required")
            durations = [iter(profile) for profile in profiles]
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(len(self.counts) + 1)]
        arrival_pos = 0

        def next_arrival(now):
            nonlocal arrival_pos
            if replay is None:
                return now + streams[0].exponential(1 / self.l)
            if arrival_pos == len(replay):
                return math.inf
            value = replay[arrival_pos]
            arrival_pos += 1
            return value

        queue = deque()
        free = [list(range(n)) for n in self.counts]
        completions = []
        order = sorted(range(len(self.counts)), key=lambda i: (self.thresholds[i], i))
        arrived = completed = started = events = 0
        busy = np.zeros(len(self.counts), dtype=int)
        busy_area = np.zeros(len(self.counts))
        customer_starts = np.zeros(len(self.counts), dtype=int)
        batch_starts = np.zeros(len(self.counts), dtype=int)
        cohort_arrivals, cohort_waits, cohort_responses = [], [], []
        queue_area = population_area = total_queue_area = total_population_area = 0.0
        started_exposure = completed_residence = 0.0
        now = 0.0
        arrival = next_arrival(now)
        trace = []

        def dispatch(t):
            nonlocal started, started_exposure
            for sigma in order:
                while free[sigma] and len(queue) >= self.thresholds[sigma]:
                    server = heapq.heappop(free[sigma])
                    batch = [queue.popleft() for _ in range(min(len(queue), self.capacities[sigma]))]
                    if durations is None:
                        duration = streams[sigma + 1].exponential(1 / self.rates[sigma])
                    else:
                        try:
                            duration = next(durations[sigma])
                        except StopIteration as exc:
                            raise ValueError(f"service replay exhausted for type {sigma}") from exc
                    if not math.isfinite(t + duration) or t + duration <= t:
                        raise ArithmeticError("service clock cannot advance at this numerical scale")
                    heapq.heappush(completions, (t + duration, sigma, server, batch))
                    busy[sigma] += 1
                    started += len(batch)
                    started_exposure += sum(t - a for a, _ in batch)
                    if t >= warmup:
                        batch_starts[sigma] += 1
                        customer_starts[sigma] += len(batch)
                    for a, cid in batch:
                        if cid is not None:
                            cohort_waits[cid] = t - a
                    if record_trace:
                        trace.append(
                            {"time": t, "type": sigma, "server": server, "batch": len(batch), "queue": len(queue)}
                        )

        while True:
            finish = completions[0][0] if completions else math.inf
            t = min(arrival, finish, horizon)
            dt = t - now
            observed = max(0.0, t - max(now, warmup))
            total_queue_area += len(queue) * dt
            total_population_area += (arrived - completed) * dt
            queue_area += len(queue) * observed
            population_area += (arrived - completed) * observed
            busy_area += busy * observed
            now = t
            if t >= horizon:
                break
            events += 1
            if events > max_events:
                raise RuntimeError("event budget exceeded; no partial result returned")
            if finish <= arrival:
                _, sigma, server, batch = heapq.heappop(completions)
                busy[sigma] -= 1
                heapq.heappush(free[sigma], server)
                completed += len(batch)
                completed_residence += sum(t - a for a, _ in batch)
                for a, cid in batch:
                    if cid is not None:
                        cohort_responses[cid] = t - a
            else:
                cid = None
                if warmup <= t < cohort_end:
                    cid = len(cohort_arrivals)
                    cohort_arrivals.append(t)
                    cohort_waits.append(None)
                    cohort_responses.append(None)
                queue.append((t, cid))
                arrived += 1
                arrival = next_arrival(t)
                if arrival <= t and replay is None:
                    raise ArithmeticError("arrival clock cannot advance at this numerical scale")
            dispatch(t)
        unfinished_wait = sum(horizon - a for a, _ in queue)
        unfinished_residence = unfinished_wait + sum(horizon - a for *_, batch in completions for a, _ in batch)
        size = len(cohort_arrivals)
        # Unknown delays necessarily exceed every requested tail time: their
        # arrival precedes cohort_end, hence follow-up is at least max(tails).
        waiting_tails = [sum(w is None or w > t for w in cohort_waits) / size for t in tails] if size else []
        response_tails = [sum(v is None or v > t for v in cohort_responses) / size for t in tails] if size else []
        observation = horizon - warmup
        result = HeterogeneousBulkSimResults(
            arrived=arrived,
            completed=completed,
            unfinished=arrived - completed,
            started=started,
            observation_time=observation,
            queue_area=queue_area,
            population_area=population_area,
            total_queue_area=total_queue_area,
            total_population_area=total_population_area,
            started_wait_exposure=started_exposure,
            unfinished_wait_exposure=unfinished_wait,
            completed_residence=completed_residence,
            unfinished_residence=unfinished_residence,
            waiting_proxy=queue_area / (observation * self.l),
            response_proxy=population_area / (observation * self.l),
            busy_fractions=(busy_area / observation / np.array(self.counts)).tolist(),
            customer_throughputs=(customer_starts / observation).tolist(),
            batch_throughputs=(batch_starts / observation).tolist(),
            cohort_size=size,
            cohort_unstarted=cohort_waits.count(None),
            cohort_unfinished=cohort_responses.count(None),
            tail_times=tails,
            waiting_tails=waiting_tails,
            response_tails=response_tails,
            iteration_trace=trace,
        )
        self._set_duration(result, start)
        return result
