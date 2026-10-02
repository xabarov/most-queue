"""Offline synthetic Acme timestamp, resource and observability contracts."""

import csv
import io
from datetime import datetime, timezone

import pytest

from most_queue.sim.utils.acme_trace import ACME_FIELDS, acme_snapshot, parse_acme_kalos


def record(idx="a", submit=0, start=2, end=5, need=1, state="COMPLETED", **overrides):
    def time(value):
        return "" if value is None else datetime.fromtimestamp(1700000000 + value, timezone.utc).isoformat()

    row = dict.fromkeys(ACME_FIELDS, "")
    row.update(
        job_id=idx,
        user="synthetic",
        node_num="1",
        gpu_num=str(need),
        cpu_num="1",
        mem_per_pod_GB="1",
        type="Eval",
        state=state,
        submit_time=time(submit),
        start_time=time(start),
        end_time=time(end),
        duration=str(end - submit if end is not None else 5),
        queue=str(start - submit if start is not None else 0),
        gpu_time=str(need * (end - submit) if end is not None else 5),
    )
    row.update(overrides)
    return row


def parse(rows, capacity=8):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=ACME_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return parse_acme_kalos(stream.getvalue().splitlines(), capacity)


def test_execution_is_not_duration_or_gpu_time_and_counts_partition_source():
    source = parse(
        [
            record(),
            record("b", state="CANCELLED"),
            record("c", state="FAILED"),
            record("d", need=0),
            record("e", end=None, state="RUNNING"),
            record("f", end=2, state="FAILED"),
        ]
    )
    assert source.jobs[0].runtime == 3 and source.jobs[0].wait == 2
    assert len(source.jobs) == 1 and len(source.terminal_jobs) == 3
    assert source.audit["derived_fields"]["duration_equals_sojourn"] == 5
    assert source.audit["derived_fields"]["duration_not_execution"] == 5
    assert source.audit["counts"] == {
        "records": 6,
        "accepted_completed": 1,
        "accepted_cancelled": 1,
        "accepted_failed": 1,
        "cpu_only": 1,
        "nonterminal_state": 1,
        "nonpositive_execution": 1,
    }
    assert sum(source.audit["counts"].values()) == 12


@pytest.mark.parametrize(
    "state,outcome",
    [("CANCELLED", "cancelled"), ("FAILED", "failed"), ("TIMEOUT", "timed_out"), ("NODE_FAIL", "node_failed")],
)
def test_terminal_labels_not_conflated_or_used_for_fitting(state, outcome):
    source = parse([record(), record("b", state=state)])
    assert [j.outcome for j in source.terminal_jobs] == ["completed", outcome]
    assert len(source.jobs) == 1
    assert not hasattr(source.terminal_jobs[1], "runtime_limit")


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"need": 0.5}, "unsupported_gpu_request"),
        ({"need": -1}, "unsupported_gpu_request"),
        ({"need": 9}, "unsupported_gpu_request"),
        ({"node_num": "0.5"}, "invalid_node_request"),
        ({"start": None}, "unobserved_execution"),
        ({"end": None}, "unobserved_execution"),
        ({"start": -1}, "negative_wait"),
        ({"end": 1}, "nonpositive_execution"),
        ({"state": "PENDING", "end": None}, "nonterminal_state"),
    ],
)
def test_audited_exclusions_not_rounded_or_imputed(changes, reason):
    source = parse([record(), record("b", **changes)])
    assert source.audit["counts"][reason] == 1 and len(source.terminal_jobs) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "UNKNOWN"},
        {"gpu_num": "nan"},
        {"node_num": "inf"},
        {"duration": ""},
        {"queue": "bad"},
        {"gpu_time": "nan"},
        {"submit_time": ""},
        {"submit_time": "2023-01-01"},
        {"start_time": "yesterday"},
        {"end_time": "bad"},
        {"submit_time": "1960-01-01T00:00:00+00:00"},
    ],
)
def test_malformed_fields_fail(changes):
    with pytest.raises(ValueError):
        parse([record(**changes)])


def test_timezone_offsets_are_normalized_and_ties_preserve_source_order():
    first = record()
    second = record("b", submit_time="2023-11-15T06:13:20+08:00")
    source = parse([first, second])
    assert source.jobs[0].submit == source.jobs[1].submit
    assert [j.job_id for j in source.jobs] == ["a", "b"]


@pytest.mark.parametrize(
    "rows",
    [[], [record(state="FAILED")], [record(), record()], [record(), record("b", submit=-1)], [record(job_id="")]],
)
def test_empty_completed_duplicate_and_unsorted_rejected(rows):
    with pytest.raises(ValueError):
        parse(rows)


@pytest.mark.parametrize("header", ["job_id", ",".join(ACME_FIELDS[:-1]), ",".join((*ACME_FIELDS[:-1], "queue"))])
def test_schema_fail_closed(header):
    with pytest.raises(ValueError, match="17 unique"):
        parse_acme_kalos([header])


def test_malformed_csv_width():
    with pytest.raises(ValueError, match="record width"):
        parse_acme_kalos([",".join(ACME_FIELDS), "a,b"])


def test_snapshot_exact_boundaries_and_unsuccessful_labels():
    source = parse(
        [
            record("a", 0, 0, 5),
            record("b", 1, 2, 10),
            record("c", 2, 5, 6),
            record("d", 3, 7, 8),
            record("e", 4, 8, 9, state="FAILED"),
            record("f", 5, 5, 6),
        ]
    )
    running, waiting = acme_snapshot(source, 1700000005)
    assert [j.job_id for j in running] == ["b", "c"]
    assert [j.job_id for j in waiting] == ["d"]
    assert [j.job_id for j in acme_snapshot(source, 1700000005, include_unsuccessful=True)[1]] == ["d", "e"]


@pytest.mark.parametrize("at", [-1, True, float("nan"), float("inf")])
def test_invalid_snapshot_time(at):
    with pytest.raises(ValueError):
        acme_snapshot(parse([record()]), at)


def test_overcapacity_snapshot_never_clipped():
    source = parse([record(need=8), record("b")])
    with pytest.raises(ValueError, match="exceeds capacity"):
        acme_snapshot(source, 1700000003)
