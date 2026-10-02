"""Empirical marked arrivals with zero gaps and explicit matched controls.

Circular blocks use the end-to-start convention documented at
https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html.
This is a workload generator, not a stationarity assumption or population CI.
"""

from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np

from most_queue.random.trace_resampling import circular_block_indices


def _count(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


@dataclass(frozen=True)
class ArrivalMark:
    """A preceding gap in seconds, resource demand, and a coupled feature bundle."""

    gap: float
    need: int
    context: str | None = None
    requested_time: float | None = None

    def __post_init__(self):
        _real(self.gap, "gap")
        _count(self.need, "need")
        if self.context is not None and (not isinstance(self.context, str) or not self.context):
            raise ValueError("context must be a nonempty string or None")
        if self.requested_time is not None:
            _real(self.requested_time, "requested_time")
            if self.requested_time == 0:
                raise ValueError("requested_time must be positive or None")


@dataclass(frozen=True)
class MarkedArrivalBootstrap:
    """Immutable donor tuples; callers must supply a temporally valid history."""

    donors: tuple[ArrivalMark, ...]

    def __post_init__(self):
        object.__setattr__(self, "donors", tuple(self.donors))
        if not self.donors or any(not isinstance(item, ArrivalMark) for item in self.donors):
            raise ValueError("donors must be nonempty ArrivalMark records")

    @classmethod
    def fit(cls, arrivals, needs, contexts, requested_times=None):
        """Form true adjacent gaps; the first row supplies only a predecessor.

        Equal timestamps are retained without jitter. Filtering history before
        calling this method changes adjacency and is the caller's responsibility.
        """
        arrivals, needs, contexts = tuple(arrivals), tuple(needs), tuple(contexts)
        requests = (None,) * len(arrivals) if requested_times is None else tuple(requested_times)
        if len(arrivals) < 2 or not len(arrivals) == len(needs) == len(contexts) == len(requests):
            raise ValueError("history fields must match and contain at least two rows")
        times = np.array([_real(value, "arrival") for value in arrivals])
        gaps = np.diff(times)
        if np.any(gaps < 0):
            raise ValueError("arrivals must be chronological")
        # Validate the predecessor's fields too; it is not itself a donor.
        records = tuple(
            ArrivalMark(float(gap), need, context, request)
            for gap, need, context, request in zip(np.r_[0, gaps], needs, contexts, requests)
        )
        return cls(records[1:])

    def sample(self, uniforms, *, block_length=1):
        """Select uniform donor starts, wrap fixed blocks, and truncate to size."""
        indices = circular_block_indices(len(self.donors), uniforms, block_length)
        return tuple(self.donors[int(index)] for index in indices)


def anchored_permutation(uniforms, warmup):
    """Shuffle each cohort partition with its first row fixed, using stable keys.

    Keeping rows 0 and warmup fixed preserves the first arrival and measured
    arrival horizon when whole gap/mark/service tuples move together.
    """
    values = np.asarray(uniforms)
    if values.ndim != 1 or not values.size or values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError("uniform keys must be a nonempty finite numeric vector")
    if np.any((values <= 0) | (values >= 1)):
        raise ValueError("uniform keys must lie strictly inside (0,1)")
    warmup = _count(warmup, "warmup", minimum=0)
    if warmup >= len(values):
        raise ValueError("warmup must leave measured jobs")
    order = np.arange(len(values))
    for begin, end in ((0, warmup), (warmup, len(values))):
        if end > begin + 1:
            order[begin + 1 : end] = begin + 1 + np.argsort(values[begin + 1 : end], kind="stable")
    return order


def decouple_gaps(records, order):
    """Retain each gap while moving coupled need/context/request bundles."""
    records, order = tuple(records), np.asarray(order)
    if order.ndim != 1 or order.dtype.kind not in "iu" or sorted(order.tolist()) != list(range(len(records))):
        raise ValueError("order must be a complete permutation of records")
    if not records or any(not isinstance(item, ArrivalMark) for item in records):
        raise ValueError("records must be nonempty ArrivalMark records")
    return tuple(replace(records[int(index)], gap=record.gap) for record, index in zip(records, order))


def arrival_times(records):
    """Accumulate nonnegative gaps, anchoring the first job at zero."""
    records = tuple(records)
    if not records or any(not isinstance(item, ArrivalMark) for item in records):
        raise ValueError("records must be nonempty ArrivalMark records")
    with np.errstate(over="ignore"):
        values = np.cumsum([record.gap for record in records]) - records[0].gap
    if not np.all(np.isfinite(values)):
        raise ValueError("cumulative arrival timestamps overflow")
    return values
