"""SWF ingestion and information-boundary tests, with synthetic records only."""

from dataclasses import replace

import pytest

from most_queue.sim.utils.workload_trace import SwfJob, SwfTrace, chronological_split, parse_swf


def record(job_id=1, submit=0, wait=0, runtime=2, need=1, **changes):
    """Make one synthetic 18-field record without licensed trace data."""
    values = [job_id, submit, wait, runtime, need, -1, -1, need, 10, -1, 1, -1, -1, -1, -1, -1, -1, -1]
    for index, value in changes.items():
        values[int(index)] = value
    return " ".join(str(value) for value in values)


def test_wait_is_not_service_and_tie_order_survives():
    trace = parse_swf(["; MaxProcs: 4", record(2, wait=10), record(1)], 4)
    assert [job.job_id for job in trace.jobs] == [2, 1]
    assert trace.jobs[0].completed_at == 12
    assert trace.jobs[0].runtime == 2
    assert trace.audit["counts"] == {"records": 2, "accepted": 2}


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"10": 5, "3": -1}, "not_completed_status"),
        ({"3": 0}, "nonpositive_runtime"),
        ({"2": -1}, "unknown_wait"),
        ({"4": 5}, "invalid_allocation"),
        ({"7": 2}, "requested_allocation_mismatch"),
        ({"16": 0}, "explicit_predecessor"),
    ],
)
def test_mutually_exclusive_audit(change, reason):
    trace = parse_swf([record(), record(2, **change)], 4)
    assert len(trace.jobs) == 1
    assert trace.audit["counts"] == {"records": 2, "accepted": 1, reason: 1}


@pytest.mark.parametrize(
    "bad",
    [
        "1 2",
        record(**{"3": "nan"}),
        record(**{"6": "inf"}),
        record(**{"4": 1.5}),
        record(**{"0": 0}),
        record(**{"1": -1}),
        record(**{"10": 1.5}),
        record(**{"1": "html"}),
    ],
)
def test_malformed_fails_closed(bad):
    with pytest.raises(ValueError):
        parse_swf([bad], 4)


@pytest.mark.parametrize(
    "lines",
    [
        [record(), record()],
        [record(1, submit=2), record(2, submit=1)],
        ["; Preemption: TS", record()],
        ["; MaxProcs: 8", record()],
        ["; MaxProcs: 4", "; MaxProcs: 8", record()],
        ["; comment only"],
    ],
)
def test_incompatible_logs_rejected(lines):
    with pytest.raises(ValueError):
        parse_swf(lines, 4)


@pytest.mark.parametrize("capacity", [True, 0, -1, 1.5])
def test_bad_capacity(capacity):
    with pytest.raises(ValueError):
        parse_swf([record()], capacity)


def test_cutoff_excludes_future_completions_and_exact_ties():
    jobs = (
        SwfJob(1, 0, 0, 2, 1),
        SwfJob(2, 1, 0, 5, 1),
        SwfJob(3, 2, 4, 1, 1),
        SwfJob(4, 6, 0, 1, 1),
        SwfJob(5, 10, 0, 1, 1),
    )
    trace = SwfTrace(jobs, 4, {}, (0, 10))
    training, heldout, audit = chronological_split(trace)
    assert training == jobs[:1]
    assert heldout == jobs[3:]
    assert audit["unfinished_at_cutoff_excluded"] == 2
    changed = replace(trace, jobs=jobs[:3] + tuple(replace(job, runtime=1000) for job in jobs[3:]))
    assert chronological_split(changed)[0] == training


@pytest.mark.parametrize("fraction", [0, 1, -1, float("nan"), float("inf")])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        chronological_split(parse_swf([record(), record(2, submit=10)], 4), fraction)
