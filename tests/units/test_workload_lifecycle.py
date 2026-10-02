"""Synthetic SWF observability and exact snapshot-boundary tests."""

import pytest

from most_queue.sim.utils.workload_lifecycle import observed_snapshot, parse_swf_lifecycle


def row(idx=1, submit=0, wait=0, runtime=1, need=1, request=10, status=1):
    return f"{idx} {submit} {wait} {runtime} {need} -1 -1 {need} {request} -1 {status} -1 -1 -1 -1 -1 -1 -1"


def test_cancelled_positive_runtime_is_retained_but_not_fitted_as_completion():
    source = parse_swf_lifecycle([row(), row(2, runtime=12, status=5)], 2)
    assert len(source.jobs) == 2 and len(source.completed.jobs) == 1
    assert source.jobs[1].status == 5 and source.jobs[1].runtime == 12
    assert source.audit["requests"]["status_5_exceeded"] == 1


def test_unknown_and_zero_cancelled_runtime_are_not_zero_work_jobs():
    source = parse_swf_lifecycle([row(), row(2, runtime=-1, status=5), row(3, runtime=0, status=5)], 2)
    assert len(source.jobs) == 1
    assert source.audit["counts"]["unobserved_or_zero_runtime"] == 2


@pytest.mark.parametrize("requested_time", [0, -1])
def test_nonpositive_request_is_unavailable_not_timeout(requested_time):
    source = parse_swf_lifecycle([row(request=requested_time)], 2)
    assert source.jobs[0].requested_time is None


def test_snapshot_exact_time_boundaries_and_future_job_exclusion():
    rows = [
        row(1, 0, 0, 5),  # ended exactly at boundary
        row(2, 1, 1, 8),  # running, three seconds elapsed
        row(3, 2, 3, 1),  # start exactly at boundary is running
        row(4, 3, 4, 1),  # pending
        row(5, 4, 4, 1, status=5),  # pending cancellation
        row(6, 5, 0, 1),  # arriving at boundary: not carry
    ]
    source = parse_swf_lifecycle(rows, 2)
    initial = observed_snapshot(source, 5)
    assert [j.job_id for j in initial.running] == [2, 3]
    assert [j.job_id for j in initial.waiting] == [4]
    assert [j.job_id for j in observed_snapshot(source, 5, include_cancelled=True).waiting] == [4, 5]


def test_overallocated_snapshot_fails_instead_of_clipping():
    source = parse_swf_lifecycle([row(1, runtime=10, need=2), row(2, runtime=10)], 2)
    with pytest.raises(ValueError, match="exceeds capacity"):
        observed_snapshot(source, 1)


@pytest.mark.parametrize("at", [-1, float("nan"), float("inf")])
def test_invalid_snapshot_time(at):
    with pytest.raises(ValueError):
        observed_snapshot(parse_swf_lifecycle([row()], 2), at)


@pytest.mark.parametrize("record", ["1 2", row(status=1.5), row(request="nan"), row(1, runtime=1e308, wait=1e308)])
def test_malformed_source_is_still_rejected(record):
    with pytest.raises(ValueError):
        parse_swf_lifecycle([record], 2)
