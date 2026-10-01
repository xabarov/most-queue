"""Reservation-preserving conservative backfilling for homogeneous resources.

Feitelson and Mu'alem Weil, Utilization and Predictability in Scheduling the
IBM SP2 with Backfilling (1998), section 2.1. This calendar receives forecasts
only; actual job durations and completion events are deliberately inaccessible.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Reservation:
    """A half-open interval reserving a number of interchangeable servers."""

    start: float
    end: float
    need: int


def first_fit(capacity, profile, need, duration, now):
    """Find the first interval with enough capacity throughout its duration."""
    changes = defaultdict(int)
    changes[now] = 0
    for slot in profile.values():
        changes[max(now, slot.start)] += slot.need
        changes[slot.end] -= slot.need
    used, anchor = 0, now
    for when, delta in sorted(changes.items()):
        if anchor is not None and anchor + duration <= when:
            break
        used += delta
        if used + need > capacity:
            anchor = None
        elif anchor is None:
            anchor = when
    if anchor is None or not np.isfinite(anchor + duration):
        raise ValueError("reservation time overflows; rescale the trace")
    if anchor + duration <= anchor:
        raise ValueError("duration is below timestamp precision; rescale the trace")
    return Reservation(anchor, anchor + duration, need)


def compress_reservations(capacity, active, requests, previous, now):
    """Compress existing reservations in profile order, then insert new jobs.

    ``requests`` maps arrival-ordered IDs to (need, estimated duration).
    Remove/reinsert ONE old job at a time, keeping all other reservations.
    A valid old slot is always feasible, hence compression cannot postpone it.
    The caller must invalidate the calendar when an active forecast overruns.
    """
    profile = {job.idx: Reservation(now, job.estimate_end, job.need) for job in active}
    profile.update(previous)
    for idx in sorted(previous, key=lambda key: (previous[key].start, key)):
        old = profile.pop(idx)
        need, duration = requests[idx]
        slot = first_fit(capacity, profile, need, duration, now)
        if slot.start > old.start:
            raise RuntimeError("a valid conservative reservation was postponed")
        profile[idx] = slot
    for idx, (need, duration) in requests.items():
        if idx not in previous:
            profile[idx] = first_fit(capacity, profile, need, duration, now)
    return {idx: profile[idx] for idx in requests}
