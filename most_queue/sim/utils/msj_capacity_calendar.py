"""Opt-in time-varying capacity for MSJ replay, with an explicit known domain.

Capacity is piecewise-constant and right-continuous: ``capacities[i]`` holds on
``[times[i], times[i + 1])``, and the last value holds on ``[times[-1], domain_end]``.
There is no forward or backward extrapolation: any query outside
``[times[0], domain_end]`` is a caller error, not an assumed constant.
"""

from bisect import bisect_right
from dataclasses import dataclass
from numbers import Real

import numpy as np


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _nonnegative_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class CapacityCalendar:
    """A pinned, explicitly-bounded capacity step function.

    ``times`` must be strictly increasing and finite; ``capacities`` are
    nonnegative integers, one per breakpoint (zero is a genuine drained-to-
    nothing day, not a placeholder). ``domain_end`` must exceed the last
    breakpoint, so the final interval is nonempty.
    """

    times: tuple[float, ...]
    capacities: tuple[int, ...]
    domain_end: float

    def __post_init__(self):
        if not self.times or len(self.times) != len(self.capacities):
            raise ValueError("times and capacities must be the same nonempty length")
        times = tuple(_finite_real(t, "breakpoint time") for t in self.times)
        capacities = tuple(_nonnegative_int(c, "capacity") for c in self.capacities)
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("breakpoint times must be strictly increasing")
        domain_end = _finite_real(self.domain_end, "domain_end")
        if domain_end <= times[-1]:
            raise ValueError("domain_end must exceed the last breakpoint time")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "capacities", capacities)
        object.__setattr__(self, "domain_end", domain_end)

    @property
    def start(self) -> float:
        """The first instant with a known capacity."""
        return self.times[0]

    def capacity_at(self, instant) -> int:
        """Return the capacity holding at ``instant``; raise outside the domain."""
        instant = _finite_real(instant, "instant")
        if not self.start <= instant <= self.domain_end:
            raise ValueError("instant is outside the capacity calendar's known domain")
        idx = bisect_right(self.times, instant) - 1
        return self.capacities[max(0, idx)]

    def max_capacity(self) -> int:
        """The highest capacity ever available within the domain."""
        return max(self.capacities)

    def breakpoints_between(self, begin, end) -> list[float]:
        """Breakpoint instants in ``(begin, end]``, clipped to the domain."""
        begin, end = _finite_real(begin, "begin"), _finite_real(end, "end")
        if end < begin:
            raise ValueError("end must not precede begin")
        lo = bisect_right(self.times, begin)
        return [t for t in self.times[lo:] if t <= min(end, self.domain_end)]


def constant_calendar(capacity, domain_start, domain_end) -> CapacityCalendar:
    """A single-breakpoint calendar; the fixed-pool control scenario."""
    return CapacityCalendar((_finite_real(domain_start, "domain_start"),), (capacity,), domain_end)


def daily_calendar(daily_capacity: dict, day_seconds: float = 86400.0) -> CapacityCalendar:
    """Build a calendar from ``{day_index: capacity}`` covering consecutive days.

    ``day_index`` values need not be contiguous; a gap is a genuine unknown
    period and raises rather than being silently bridged by an adjacent value.
    """
    if not daily_capacity:
        raise ValueError("daily_capacity must not be empty")
    days = sorted(daily_capacity)
    if days[-1] - days[0] + 1 != len(days):
        raise ValueError("daily_capacity has missing days; no forward/backfill is allowed")
    times = tuple(day * day_seconds for day in days)
    capacities = tuple(daily_capacity[day] for day in days)
    domain_end = (days[-1] + 1) * day_seconds
    return CapacityCalendar(times, capacities, domain_end)
