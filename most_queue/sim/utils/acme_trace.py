"""Audited Acme/Kalos ingestion, without treating queue time as service.

Schema: https://github.com/InternLM/AcmeTrace. The pinned release's duration
column includes waiting, despite its documentation. Only end-start is used.
GPU demand is a request, not measured utilization or a placement trajectory.
"""

import csv
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from most_queue.sim.utils.workload_trace import positive_integer

ACME_FIELDS = (
    "job_id",
    "user",
    "node_num",
    "gpu_num",
    "cpu_num",
    "mem_per_pod_GB",
    "shared_mem_per_pod",
    "type",
    "state",
    "submit_time",
    "start_time",
    "end_time",
    "fail_time",
    "stop_time",
    "duration",
    "queue",
    "gpu_time",
)
ACME_OUTCOMES = {
    "COMPLETED": "completed",
    "CANCELLED": "cancelled",
    "FAILED": "failed",
    "TIMEOUT": "timed_out",
    "NODE_FAIL": "node_failed",
}


@dataclass(frozen=True)
class AcmeJob:
    """One positive observed execution interval with an unmodified terminal label."""

    job_id: str
    submit: float
    started_at: float
    ended_at: float
    need: int
    outcome: str
    workload_type: str

    @property
    def runtime(self):
        """Occupied execution interval, never submit-to-end latency."""
        return self.ended_at - self.started_at

    @property
    def wait(self):
        """Historical wait used only for retrospective state diagnostics."""
        return self.started_at - self.submit

    @property
    def completed_at(self):
        """Time the outcome becomes observable; success is a separate label."""
        return self.ended_at


@dataclass(frozen=True)
class AcmeTrace:
    """Completed fitting cohort plus all eligible labelled GPU execution intervals.

    jobs contains only completed outcomes for chronological_split compatibility.
    terminal_jobs also includes non-successes; unfinished demand is not imputed.
    """

    jobs: tuple[AcmeJob, ...]
    terminal_jobs: tuple[AcmeJob, ...]
    capacity: int
    submit_range: tuple[float, float]
    audit: dict


def _time(value, field):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone is required")
        return parsed.timestamp()
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid timezone-aware {field}") from exc


def _number(value, field, *, optional=False):
    if not value and optional:
        return None
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid numeric {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {field}")
    return result


def parse_acme_kalos(lines: Iterable[str], capacity=2416) -> AcmeTrace:
    """Validate the Kalos schema, preserve chronological ties and audit exclusions.

    Malformed fields, duplicate IDs, unknown states and out-of-order submits
    fail loudly. Well-formed but unobservable/unsupported jobs are counted, not
    zero-imputed. Derived columns are audited independently of cohort selection.
    Integer GPU requests are preserved exactly; CPU-only/fractional demand is
    excluded from this scalar, homogeneous, non-sharing abstraction.
    """
    capacity = positive_integer(capacity, "capacity")
    reader = csv.DictReader(lines)
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(ACME_FIELDS)
        or set(reader.fieldnames) != set(ACME_FIELDS)
    ):
        raise ValueError("expected the 17 unique Acme/Kalos columns")
    counts, states, checks, kinds = Counter(), Counter(), Counter(), Counter()
    jobs, terminal, submits, seen = [], [], [], set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("incorrect CSV record width")
        idx = row["job_id"]
        if not idx or idx in seen:
            raise ValueError("duplicate or empty job ID")
        seen.add(idx)
        state = row["state"]
        if state not in (*ACME_OUTCOMES, "RUNNING", "PENDING"):
            raise ValueError("unknown Acme terminal/state label")
        submit, start, end = (_time(row[field], field) for field in ("submit_time", "start_time", "end_time"))
        if submit is None or submit < 0 or (submits and submit < submits[-1]):
            raise ValueError("submit must be nonnegative and chronological")
        submits.append(submit)
        need, nodes = (_number(row[field], field) for field in ("gpu_num", "node_num"))
        duration, wait = (_number(row[field], field) for field in ("duration", "queue"))
        gpu_time = _number(row["gpu_time"], "gpu_time", optional=True)
        counts["records"] += 1
        states[state] += 1
        kinds[row["type"]] += 1
        if start is not None:
            checks["queue_checked"] += 1
            checks["queue_mismatch"] += int(wait != start - submit)
        if end is not None and start is not None:
            checks["closed_intervals"] += 1
            checks["duration_not_execution"] += int(duration != end - start)
            checks["duration_equals_sojourn"] += int(duration == end - submit)
            if gpu_time is not None:
                checks["gpu_time_checked"] += 1
                checks["gpu_time_not_execution_work"] += int(gpu_time != need * (end - start))
        reasons = (
            (need == 0, "cpu_only"),
            (need < 0 or need != int(need) or need > capacity, "unsupported_gpu_request"),
            (nodes < 1 or nodes != int(nodes) or need > 8 * nodes, "invalid_node_request"),
            (state not in ACME_OUTCOMES, "nonterminal_state"),
            (start is None or end is None, "unobserved_execution"),
            (start is not None and start < submit, "negative_wait"),
            (start is not None and end is not None and end <= start, "nonpositive_execution"),
        )
        reason = next((label for bad, label in reasons if bad), None)
        if reason:
            counts[reason] += 1
            continue
        job = AcmeJob(idx, submit, start, end, int(need), ACME_OUTCOMES[state], row["type"])
        terminal.append(job)
        counts[f"accepted_{job.outcome}"] += 1
        if state == "COMPLETED":
            jobs.append(job)
    if not jobs:
        raise ValueError("no eligible completed GPU jobs")
    return AcmeTrace(
        tuple(jobs),
        tuple(terminal),
        capacity,
        (submits[0], submits[-1]),
        {"counts": dict(counts), "states": dict(states), "derived_fields": dict(checks), "workload_types": dict(kinds)},
    )


def acme_snapshot(source: AcmeTrace, at: float, *, include_unsuccessful=False):
    """Return partial retrospective running/waiting sets at a strict submit cut.

    Historical start==at is running; end==at is already released; submit==at is
    a new arrival. Missing end intervals are not fabricated. Overcapacity fails.
    """
    if isinstance(at, bool) or not math.isfinite(at) or at < 0:
        raise ValueError("snapshot time must be finite and nonnegative")
    eligible = source.terminal_jobs if include_unsuccessful else source.jobs
    carry = tuple(job for job in eligible if job.submit < at < job.ended_at)
    running = tuple(job for job in carry if job.started_at <= at)
    waiting = tuple(job for job in carry if job.started_at > at)
    if sum(job.need for job in running) > source.capacity:
        raise ValueError("observed requested-GPU snapshot exceeds capacity")
    return running, waiting
