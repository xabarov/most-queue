"""Independent unbounded FCFS batch DES for Xu et al., arXiv:2301.12865v3.

Actual timestamps and deterministic completions; no finite transition kernel.
Fixed horizon, no drain, no completed-only censored latency moments.
Energy is charged at batch START, matching the source impulse-cost objective.
"""

import math
import time
from collections import deque
from numbers import Integral, Real

import numpy as np

from most_queue.structs import BatchSmdpSimResults


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _count(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class BatchSmdpSim:
    """Decision epochs are completions and arrivals while idle.

    A finite policy contains actions for 0,...,N and a separate overflow action
    used whenever actual population>N. A callable instead sees actual population.
    Action zero waits for the next arrival, not an arbitrary timeout.
    """

    def __init__(self, seed=42, *, delay_weight=1.0, energy_weight=1.0):
        self.seed = _count(seed, "seed")
        self.delay_weight = _number(delay_weight, "delay_weight")
        self.energy_weight = _number(energy_weight, "energy_weight")
        self.rate = None
        self.times = None
        self.energies = None

    def set_sources(self, rate):
        """Set positive Poisson rate; replay uses the same objective normalization."""
        self.rate = _number(rate, "rate", positive=True)
        if not math.isfinite(1 / self.rate):
            raise ValueError("mean interarrival time is nonfinite; rescale")

    def set_servers(self, batch_times, batch_energies):
        """Set deterministic profiles for batch sizes 1,...,B."""
        times = [_number(x, "batch_time", positive=True) for x in batch_times]
        energies = [_number(x, "batch_energy") for x in batch_energies]
        if not times or len(times) != len(energies) or len(times) > 128:
            raise ValueError("equal nonempty time/energy profiles with <=128 batches required")
        self.times, self.energies = times, energies

    def run(  # pylint: disable=too-many-arguments
        self, horizon, policy, *, warmup=0.0, arrivals=None, initial_jobs=0, record_trace=False, max_events=2_000_000
    ):
        """Observe [warmup,horizon); counters include the entire [0,horizon).

        Replay times are nondecreasing in [0,horizon). At a simultaneous boundary,
        completion and all arrivals are processed before choosing the next batch.
        Runs restart their seeded RNG; unfinished requests are explicitly counted.
        """
        if self.times is None or self.rate is None:
            raise ValueError("sources and servers must be set")
        horizon = _number(horizon, "horizon", positive=True)
        warmup = _number(warmup, "warmup")
        initial_jobs = _count(initial_jobs, "initial_jobs")
        max_events = _count(max_events, "max_events", minimum=1)
        if warmup >= horizon or initial_jobs > max_events or not isinstance(record_trace, bool):
            raise ValueError("invalid warmup, initial jobs or trace flag")
        if not callable(policy):
            policy = [_count(a, "action") for a in policy]
            n = len(policy) - 2
            if n < len(self.times) or any(a > min(s, len(self.times)) for s, a in enumerate(policy)):
                raise ValueError("finite policy must cover 0,...,N,overflow with feasible actions and N>=B")

            def decision(population):
                return policy[population if population <= n else -1]

        else:
            decision = policy
        if arrivals is not None:
            arrivals = [_number(t, "arrival time") for t in arrivals]
            if len(arrivals) > max_events or any(a > b for a, b in zip(arrivals, arrivals[1:])):
                raise ValueError("replay must be sorted and fit the event limit")
            if arrivals and arrivals[-1] >= horizon:
                raise ValueError("replay arrivals must be strictly before horizon")
        rng = np.random.default_rng(self.seed)
        arrival_index = 0
        next_arrival = (
            (arrivals[0] if arrivals else math.inf) if arrivals is not None else rng.exponential(1 / self.rate)
        )
        queue = deque([0.0] * initial_jobs)
        active = []
        completion = math.inf
        result = BatchSmdpSimResults(initial_jobs=initial_jobs, observation_time=horizon - warmup)
        observed_energy = 0.0
        busy_area = 0.0
        events = 0
        clock = 0.0
        start = time.perf_counter()

        def receive(now):
            nonlocal next_arrival, arrival_index, events
            while next_arrival == now:
                queue.append(now)
                result.arrived += 1
                events += 1
                if events > max_events:
                    raise ValueError("event limit exceeded; shorten horizon or raise max_events")
                if arrivals is None:
                    next_arrival = now + rng.exponential(1 / self.rate)
                else:
                    arrival_index += 1
                    next_arrival = arrivals[arrival_index] if arrival_index < len(arrivals) else math.inf

        receive(clock)
        while clock < horizon:
            if not active:
                population = len(queue)
                action = _count(decision(population), "policy action")
                if action > min(population, len(self.times)):
                    raise ValueError("policy chose an infeasible batch")
                if record_trace:
                    result.iteration_trace.append({"time": clock, "population": population, "action": action})
                if action:
                    active = [queue.popleft() for _ in range(action)]
                    completion = clock + self.times[action - 1]
                    if completion <= clock:
                        raise ValueError("service period below timestamp resolution; rescale")
                    result.batch_starts += 1
                    result.energy += self.energies[action - 1]
                    if clock >= warmup:
                        observed_energy += self.energies[action - 1]
            next_time = min(next_arrival, completion, horizon)
            population = len(queue) + len(active)
            result.total_population_area += population * (next_time - clock)
            observed_period = max(0.0, next_time - max(clock, warmup))
            result.population_area += population * observed_period
            if active:
                busy_area += observed_period
            clock = next_time
            if clock >= horizon:
                break
            if completion == clock:
                result.completed += len(active)
                result.completed_residence += sum(clock - entered for entered in active)
                active = []
                completion = math.inf
                events += 1
                if events > max_events:
                    raise ValueError("event limit exceeded; shorten horizon or raise max_events")
            receive(clock)
        result.unfinished = len(queue) + len(active)
        result.unfinished_residence = sum(horizon - t for t in queue) + sum(horizon - t for t in active)
        result.power = observed_energy / result.observation_time
        result.response_proxy = result.population_area / (result.observation_time * self.rate)
        result.average_cost = self.delay_weight * result.response_proxy + self.energy_weight * result.power
        result.busy_fraction = busy_area / result.observation_time
        result.duration = time.perf_counter() - start
        return result
