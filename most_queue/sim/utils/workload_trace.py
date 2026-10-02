"""Strict, audited SWF ingestion for a completed rigid-job replay cohort.

Field semantics: https://www.cs.huji.ac.il/labs/parallel/workload/swf.html.
This adapter does not reconstruct cancelled demand, preemption, dependencies,
or the original scheduler. Exclusions deliberately change the population.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SwfJob:
    """Observed rigid job; wait is historical, never a replay input."""

    job_id: int
    submit: float
    wait: float
    runtime: float
    need: int

    @property
    def completed_at(self) -> float:
        """Earliest instant at which the recorded outcome is available."""
        return self.submit + self.wait + self.runtime


@dataclass(frozen=True)
class SwfTrace:
    """Accepted cohort, mutually exclusive exclusions and raw time range."""

    jobs: tuple[SwfJob, ...]
    capacity: int
    audit: dict
    submit_range: tuple[float, float]


def positive_integer(value, name, minimum=1):
    """Validate counts without accepting booleans or truncating fractions."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def parse_swf(lines: Iterable[str], capacity: int) -> SwfTrace:
    """Read 18-column SWF, fail on malformed data, audit cohort exclusions.

    Only status=1, positive runtime, valid wait/allocation, requested=allocated,
    and no explicit predecessor are eligible. IDs must be unique even for
    excluded jobs; preemption summaries/segments require another adapter.
    Source order must be chronological; equal-submit order is preserved.
    """
    capacity = positive_integer(capacity, "capacity")
    jobs, seen, submits = [], set(), []
    counts = Counter()
    statuses = Counter()
    header = {}
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(";"):
            key, sep, value = line[1:].strip().partition(":")
            if sep and key in ("MaxProcs", "MaxNodes", "Preemption"):
                if key in header and header[key] != value.strip():
                    raise ValueError(f"conflicting {key} header")
                header[key] = value.strip()
            continue
        fields = line.split()
        try:
            values = np.array([float(v) for v in fields])
        except ValueError as exc:
            raise ValueError(f"line {number}: nonnumeric SWF record") from exc
        if len(values) != 18 or not np.all(np.isfinite(values)):
            raise ValueError(f"line {number}: expected 18 finite SWF fields")
        for idx in (0, 4, 7, 10, 16):
            if values[idx] != int(values[idx]):
                raise ValueError(f"line {number}: field {idx + 1} must be integral")
        job_id = int(values[0])
        submit, wait, runtime = values[1:4]
        if job_id <= 0 or job_id in seen:
            raise ValueError(f"line {number}: duplicate or invalid job ID")
        if submit < 0 or (submits and submit < submits[-1]):
            raise ValueError(f"line {number}: submit must be nonnegative and chronological")
        seen.add(job_id)
        submits.append(float(submit))
        counts["records"] += 1
        status = int(values[10])
        statuses[str(status)] += 1
        reasons = (
            (status != 1, "not_completed_status"),
            (runtime <= 0, "nonpositive_runtime"),
            (wait < 0, "unknown_wait"),
            (not 1 <= values[4] <= capacity, "invalid_allocation"),
            (values[7] != values[4], "requested_allocation_mismatch"),
            (values[16] != -1, "explicit_predecessor"),
        )
        reason = next((name for bad, name in reasons if bad), None)
        if reason:
            counts[reason] += 1
        else:
            job = SwfJob(job_id, float(submit), float(wait), float(runtime), int(values[4]))
            if not np.isfinite(job.completed_at):
                raise ValueError("completion timestamp overflow")
            jobs.append(job)
            counts["accepted"] += 1
    if not jobs:
        raise ValueError("no eligible completed jobs")
    declared = header.get("MaxProcs", header.get("MaxNodes"))
    if declared is not None and declared != str(capacity):
        raise ValueError("capacity differs from the SWF header")
    if header.get("Preemption", "No") != "No":
        raise ValueError("preempted/time-sliced logs require a different adapter")
    return SwfTrace(
        tuple(jobs), capacity, {"counts": dict(counts), "statuses": dict(statuses)}, (submits[0], submits[-1])
    )


def chronological_split(trace: SwfTrace, fraction=0.6):
    """Use a raw-time boundary; exclude pre-boundary unfinished outcomes.

    A held-out status=1 cohort is selected retrospectively for evaluation, but
    no held-out outcome enters fitting. Jobs completed exactly at the cutoff
    are excluded from history by the strict convention.
    """
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    first, last = trace.submit_range
    cutoff = first + fraction * (last - first)
    training = tuple(job for job in trace.jobs if job.completed_at < cutoff)
    heldout = tuple(job for job in trace.jobs if job.submit >= cutoff)
    if not training or not heldout:
        raise ValueError("split must leave both completed history and held-out arrivals")
    audit = {
        "cutoff": cutoff,
        "training": len(training),
        "heldout": len(heldout),
        "unfinished_at_cutoff_excluded": len(trace.jobs) - len(training) - len(heldout),
    }
    return training, heldout, audit
