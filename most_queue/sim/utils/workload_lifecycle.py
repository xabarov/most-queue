"""Audited observed release times, not latent completion demand for cancelled SWF jobs.

Field meanings: https://www.cs.huji.ac.il/labs/parallel/workload/swf.html.
SDSC conversion notes distinguish unobserved cancellation from positive runtime:
https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/index.html.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from most_queue.sim.utils.workload_trace import SwfJob, SwfTrace, parse_swf


@dataclass(frozen=True)
class SwfLifecycleJob(SwfJob):
    """Runtime is occupied time to the recorded terminal outcome, not latent S."""

    status: int
    requested_time: float | None

    @property
    def started_at(self):
        """Original-log start, used only for retrospective snapshot construction."""
        return self.submit + self.wait


@dataclass(frozen=True)
class SwfLifecycleTrace:
    """Completed training cohort plus observable completed/cancelled resource use."""

    completed: SwfTrace
    jobs: tuple[SwfLifecycleJob, ...]
    audit: dict


@dataclass(frozen=True)
class SwfSnapshot:
    """Partial observed state; does not recover unknown jobs or scheduler memory."""

    at: float
    running: tuple[SwfLifecycleJob, ...]
    waiting: tuple[SwfLifecycleJob, ...]


def parse_swf_lifecycle(lines, capacity):
    """Validate with the strict SWF reader; require a completed reference cohort.

    Add only status=5 jobs with positive runtime, known wait and rigid allocation.
    Unknown/zero runtimes are excluded explicitly, never replaced with zero work.
    Nonpositive requested time is unavailable, not an enforced budget.
    """
    lines = tuple(lines)
    completed = parse_swf(lines, capacity)
    jobs, counts, requests = [], Counter(), Counter()
    for line in lines:
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        values = [float(value) for value in line.split()]
        status = int(values[10])
        counts["records"] += 1
        reasons = (
            (status not in (1, 5), "other_status"),
            (values[3] <= 0, "unobserved_or_zero_runtime"),
            (values[2] < 0, "unknown_wait"),
            (not 1 <= values[4] <= capacity, "invalid_allocation"),
            (values[7] != values[4], "requested_allocation_mismatch"),
            (values[16] != -1, "explicit_predecessor"),
        )
        reason = next((name for bad, name in reasons if bad), None)
        if reason:
            counts[reason] += 1
            continue
        request = values[8] if values[8] > 0 else None
        job = SwfLifecycleJob(int(values[0]), *values[1:4], int(values[4]), status, request)
        if not np.isfinite(job.completed_at):
            raise ValueError("terminal timestamp overflow")
        jobs.append(job)
        counts[f"accepted_status_{status}"] += 1
        requests[f"status_{status}_missing"] += request is None
        requests[f"status_{status}_exceeded"] += request is not None and job.runtime > request
    return SwfLifecycleTrace(completed, tuple(jobs), {"counts": dict(counts), "requests": dict(requests)})


def observed_snapshot(source, at, *, include_cancelled=False):
    """Use submit < at, terminal > at; a start exactly at at is already running.

    Selection and residual runtime are retrospective diagnostics. Capacity is
    checked rather than silently dropping or scaling overlapping allocations.
    """
    if not np.isfinite(at) or at < 0 or not isinstance(include_cancelled, bool):
        raise ValueError("at must be finite/nonnegative and include_cancelled boolean")
    active = [
        job for job in source.jobs if job.submit < at < job.completed_at and (include_cancelled or job.status == 1)
    ]
    running = tuple(job for job in active if job.started_at <= at)
    waiting = tuple(job for job in active if job.started_at > at)
    if sum(job.need for job in running) > source.completed.capacity:
        raise ValueError("observed running allocation exceeds capacity")
    return SwfSnapshot(float(at), running, waiting)
