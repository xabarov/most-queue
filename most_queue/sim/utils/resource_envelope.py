"""Audited rigid-resource projections, not inferred quotas or node placement.

An exclusive-node reservation is an explicit counterfactual. GPU counts do not
identify quotas, available capacity, topology or physical GPU utilization.
"""

from collections import defaultdict
from dataclasses import dataclass
from numbers import Real

import numpy as np

from most_queue.sim.utils.workload_trace import positive_integer


@dataclass(frozen=True)
class ResourceRequest:
    """Known GPU and node requests under a homogeneous units-per-node assumption."""

    gpus: int
    nodes: int
    gpus_per_node: int = 8

    def __post_init__(self):
        for name in ("gpus", "nodes", "gpus_per_node"):
            positive_integer(getattr(self, name), name)
        if self.gpus > self.nodes * self.gpus_per_node:
            raise ValueError("GPU request exceeds the requested nodes' nominal inventory")

    def demand(self, allocation):
        """Project to scalar scheduler units without changing actual GPU request."""
        if allocation == "gpu_pool":
            return self.gpus
        if allocation == "exclusive_nodes":
            return self.nodes
        raise ValueError("allocation must be gpu_pool or exclusive_nodes")


def assess_envelope(capacity, arrivals, *, running=(), waiting=()):
    """Report infeasibility without dropping oversized work or resetting carry-in.

    Individual jobs must fit and the supplied initial running sum must fit.
    No total-work bound is imposed: finite queued work can drain sequentially.
    Invalid counts raise; well-formed but impossible reservations are audited.
    """
    capacity = positive_integer(capacity, "capacity")
    groups = {"running": tuple(running), "waiting": tuple(waiting), "arrivals": tuple(arrivals)}
    if not groups["arrivals"]:
        raise ValueError("at least one arrival is required")
    for values in groups.values():
        for value in values:
            positive_integer(value, "resource demand")
    oversized = {name: sum(value > capacity for value in values) for name, values in groups.items()}
    running_total = sum(groups["running"])
    reasons = []
    if any(oversized.values()):
        reasons.append("individual_demand_exceeds_capacity")
    if running_total > capacity:
        reasons.append("initial_running_exceeds_capacity")
    return {
        "capacity": capacity,
        "feasible": not reasons,
        "reasons": reasons,
        "counts": {name: len(values) for name, values in groups.items()},
        "oversized": oversized,
        "maximum_individual_demand": max(value for values in groups.values() for value in values),
        "initial_running_demand": running_total,
    }


def observed_occupancy(intervals, capacities):
    """Integrate recorded half-open reservation intervals and capacity exceedance.

    A feasible recorded interval path is a compatibility check, not an estimator
    of available capacity. Shared end/start instants are processed atomically.
    The horizon spans first start to last end, including internal idle gaps.
    """
    levels = tuple(positive_integer(value, "capacity") for value in capacities)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("capacities must be nonempty and unique")
    events = defaultdict(int)
    maximum = count = 0
    for begin, end, need in intervals:
        for value in (begin, end):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError("interval timestamps must be finite and nonnegative")
        if end <= begin:
            raise ValueError("interval must have positive duration")
        need = positive_integer(need, "resource demand")
        events[float(begin)] += need
        events[float(end)] -= need
        maximum = max(maximum, need)
        count += 1
    if not count:
        raise ValueError("at least one interval is required")
    over_time = dict.fromkeys(levels, 0.0)
    excess_work = dict.fromkeys(levels, 0.0)
    current = peak = 0
    work = 0.0
    previous = first = min(events)
    for instant, delta in sorted(events.items()):
        dt = instant - previous
        work += current * dt
        for capacity in levels:
            over_time[capacity] += dt if current > capacity else 0.0
            excess_work[capacity] += max(0, current - capacity) * dt
        current += delta
        if current < 0:
            raise RuntimeError("negative occupancy")
        peak = max(peak, current)
        previous = instant
    if current or not np.isfinite(work) or any(not np.isfinite(x) for x in excess_work.values()):
        raise ValueError("invalid or overflowing occupancy integral")
    return {
        "count": count,
        "first_start": first,
        "last_end": previous,
        "maximum_individual_demand": maximum,
        "peak_concurrent_demand": peak,
        "resource_time": work,
        "capacities": [
            {
                "capacity": c,
                "compatible_with_recorded_intervals": peak <= c,
                "seconds_above_capacity": over_time[c],
                "excess_resource_time": excess_work[c],
            }
            for c in levels
        ],
    }
