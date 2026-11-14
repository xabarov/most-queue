"""Deterministic planning, actual-count and engine-clock calibration contracts."""

# Test doubles intentionally retain the production worker signatures.
# pylint: disable=unused-argument,too-many-arguments,too-many-positional-arguments

import json
from collections import Counter

import pytest

from tools import runpod_calibration_workload as workload


@pytest.fixture(name="config")
def fixture_config():
    """Return the authorized fixed profile, without contacting a GPU."""
    return {
        "schema_version": 1,
        "input_lengths": [128, 1024, 4096],
        "output_lengths": [32, 128, 512],
        "train_repetitions": [1, 2, 3],
        "isolated_validation_repetitions": [4],
        "burst_concurrencies": [2, 4],
        "burst_repetitions": [1, 2],
        "open_loop_rates": [0.2, 0.5, 0.8],
        "open_loop_seconds": 180,
        "open_loop_seeds": [54101, 54102],
        "workload_timeout_seconds": 2700,
        "request_timeout_seconds": 120,
        "max_in_flight": 64,
        "max_requests": 1000,
    }


def test_plan_reproducible_counts_and_warmup_split(config):
    """The held-out series and warmup requests must have their intended scope."""
    plan = workload.generate_plan(config, 2026111402)
    assert plan == workload.generate_plan(config, 2026111402)
    rows = [row for series in plan for row in series["requests"]]
    counts = Counter((r["split"], r["mode"]) for r in rows)
    assert counts[("warmup", "isolated")] == 9
    assert counts[("warmup", "burst")] == 12
    assert counts[("train", "isolated")] == 27
    assert counts[("validation", "isolated")] == 9
    assert counts[("validation", "burst")] == 108
    assert len([series for series in plan if series["duration_s"] == 180]) == 6
    assert len(rows) <= 1000
    assert len(rows) == 721
    assert [r["request_index"] for r in rows] == list(range(1, len(rows) + 1))
    assert len({r["request_id"] for r in rows}) == len(rows)
    assert all((r["phase"] == "warmup") == (r["split"] == "warmup") for r in rows)
    assert {r["repetition"] for r in rows if r["split"] == "train"} == {1, 2, 3}
    assert {r["repetition"] for r in rows if r["mode"] == "isolated" and r["split"] == "validation"} == {4}


def test_plan_maximum_enforced_before_execution(config):
    """An excessive offered plan fails locally, before any request is issued."""
    with pytest.raises(ValueError, match="max_requests"):
        workload.generate_plan({**config, "max_requests": 100}, 2026111402)


def test_exact_seeded_open_loop_arrivals(config):
    """Every exponential arrival is fixed before any service completion exists."""
    import random  # pylint: disable=import-outside-toplevel

    plan = workload.generate_plan(config, 2026111402)
    for series in plan:
        if not series["duration_s"]:
            continue
        rows = series["requests"]
        rate = rows[0]["request_rate"]
        generator = random.Random(2026111402 + config["open_loop_rates"].index(rate) * 1000 + series["arrival_seed"])
        expected = []
        offset = generator.expovariate(rate)
        while offset < 180:
            expected.append(offset)
            offset += generator.expovariate(rate)
        assert [row["arrival_offset_s"] for row in rows] == expected
        assert all(row["concurrency"] is None for row in rows)


def trace_row():
    """Return one complete matching engine trace."""
    return {
        "request_id": "cmpl-mqcal-1-0",
        "queued_ts": 10.0,
        "scheduled_ts": 10.1,
        "first_token_ts": 10.2,
        "last_token_ts": 10.5,
        "arrival_time": 1777777777.0,
        "timestamp_clock": "monotonic",
        "arrival_time_clock": "wall",
        "num_generation_tokens": 32,
        "num_preemptions": 0,
    }


def client_row():
    """Return one successful usage-bearing client result."""
    return {
        "request_id": "mqcal-1",
        "server_request_id": "cmpl-mqcal-1",
        "completion_tokens": 32,
        "success": True,
        "error": None,
        "metrics": {"queue_time_ms": 100.0, "time_to_first_token_ms": 100.0, "generation_time_ms": 300.0},
    }


def test_trace_join_exact_suffix_and_metric_consistency(tmp_path):
    """Actual engine stamps are joined by exact pinned ID and duration formula."""
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(trace_row()) + "\n")
    row = workload.join_server_trace(client_row(), trace)
    assert row["success"]
    assert row["server_timestamps"] == trace_row()
    trace.write_text(json.dumps({**trace_row(), "request_id": "cmpl-mqcal-1"}) + "\n")
    assert not workload.join_server_trace(client_row(), trace)["success"]


@pytest.mark.parametrize(
    "change",
    [
        {"last_token_ts": 10.6},
        {"num_generation_tokens": 31},
        {"num_preemptions": 1},
        {"queued_ts": 0},
        {"first_token_ts": 10.7},
        {"arrival_time": float("nan")},
    ],
)
def test_trace_contract_rejects_mismatch(tmp_path, change):
    """Mismatches remain failed offered observations."""
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({**trace_row(), **change}) + "\n")
    assert not workload.join_server_trace(client_row(), path)["success"]


def test_trace_rejects_duplicate_or_wrong_server_id(tmp_path):
    """Neither duplicate finals nor another request's ID is accepted."""
    path = tmp_path / "trace.jsonl"
    path.write_text((json.dumps(trace_row()) + "\n") * 2)
    assert not workload.join_server_trace(client_row(), path)["success"]
    assert not workload.join_server_trace({**client_row(), "server_request_id": "cmpl-mqcal-2"}, path)["success"]


def test_dynamic_output_lengths_and_worker_clock(tmp_path, monkeypatch, config):
    """The child gets requested output length and its real clock is retained."""
    spec = workload.generate_plan(config, 2026111402)[0]["requests"][0]
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(trace_row()) + "\n")
    payloads = []

    class Process:
        """Fake a completed SSE reader rather than doing network I/O."""

        returncode = 0

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, payload=None, timeout=None):
            """Return realistic final SSE metrics and actual usage."""
            payloads.append(json.loads(payload))
            items = [
                {"worker_started_monotonic_s": 123.25},
                {"event": {"elapsed_s": 0.1, "chunk": {"id": "cmpl-mqcal-1", "choices": [{"text": "x"}]}}},
                {
                    "event": {
                        "elapsed_s": 0.5,
                        "chunk": {
                            "id": "cmpl-mqcal-1",
                            "usage": {"prompt_tokens": 128, "completion_tokens": 32},
                            "metrics": client_row()["metrics"],
                        },
                    }
                },
                {"event": {"elapsed_s": 0.6, "done": True, "raw_data": "[DONE]"}},
            ]
            return "\n".join(json.dumps(item) for item in items), ""

        def poll(self):
            """Indicate a completed process."""
            return 0

    monkeypatch.setattr(workload.subprocess, "Popen", Process)
    now = workload.time.monotonic()
    row = workload.execute_request(spec, "model", now + 120, 120, path, now, now)
    assert row["success"]
    assert row["client_started_monotonic_s"] == 123.25
    assert payloads[0]["max_tokens"] == 32
    assert len(payloads[0]["prompt"]) == 128
    assert payloads[0]["request_id"] == "mqcal-1"
    now = workload.time.monotonic()
    row = workload.execute_request({**spec, "output_tokens": 512}, "model", now + 120, 120, path, now, now)
    assert payloads[1]["max_tokens"] == 512
    assert not row["success"]


def test_global_deadline_retains_unsent_plan_without_offering_requests(tmp_path, config):
    """A future request is explicitly not offered once the global deadline passes."""
    plan = workload.generate_plan(config, 2026111402)
    rows = []
    result = workload.run_series(plan[0], "model", config, tmp_path / "absent", 0, rows.append)
    assert len(rows) == result["offered"] == 0
    assert result["not_offered_request_indices"] == [1]
    assert result["planned"] == 1


def test_launched_worker_timeout_retains_partial_events(tmp_path, monkeypatch, config):
    """Enforced child deadlines preserve the actual attempted request and SSE."""
    spec = workload.generate_plan(config, 2026111402)[0]["requests"][0]
    processes = []

    class Process:
        """Simulate a trickling worker that needs to be killed by the parent."""

        returncode = None

        def __init__(self, *args, **kwargs):
            self.killed = False
            processes.append(self)

        def communicate(self, payload=None, timeout=None):
            """Yield partial output after the parent's hard timeout kills us."""
            if not self.killed:
                raise workload.subprocess.TimeoutExpired("worker", timeout)
            items = [
                {"worker_started_monotonic_s": 42.5},
                {
                    "event": {
                        "elapsed_s": 0.1,
                        "raw_data": '{"choices":[{"text":"x"}]}',
                        "chunk": {"choices": [{"text": "x"}]},
                    }
                },
            ]
            return "\n".join(json.dumps(item) for item in items), ""

        def kill(self):
            """Track that timeout cancellation actually kills the child."""
            self.killed = True
            self.returncode = -9

        def poll(self):
            """Return the killed process status."""
            return self.returncode

    monkeypatch.setattr(workload.subprocess, "Popen", Process)
    now = workload.time.monotonic()
    row = workload.execute_request(spec, "model", now + 120, 120, tmp_path / "absent", now, now)
    assert processes[0].killed
    assert not row["success"]
    assert row["error"] == "Overall request deadline exceeded"
    assert row["client_started_monotonic_s"] == 42.5
    assert len(row["events"]) == 1


def test_open_loop_dispatch_does_not_wait_for_slow_service(tmp_path, monkeypatch, config):
    """Saturated client capacity produces explicit failures at fixed arrivals."""
    import threading  # pylint: disable=import-outside-toplevel
    import time  # pylint: disable=import-outside-toplevel

    rows = []
    seen = []
    blocker = threading.Event()

    def slow_request(spec, model, deadline, timeout_s, trace_path, scheduled, dispatch):
        seen.append((spec["request_index"], scheduled, dispatch))
        blocker.wait(timeout=0.08)
        return workload.failed_record(spec, scheduled, "simulated slow service", dispatch)

    monkeypatch.setattr(workload, "execute_request", slow_request)
    specs = workload.generate_plan(config, 2026111402)[0]["requests"]
    series = {
        "series_id": "fixed-arrivals",
        "duration_s": 0.03,
        "requests": [
            {**specs[0], "request_index": index + 1, "arrival_offset_s": offset}
            for index, offset in enumerate([0.0, 0.01, 0.02])
        ],
    }
    result = workload.run_series(
        series, "model", {**config, "max_in_flight": 1}, tmp_path / "absent", time.monotonic() + 1, rows.append
    )
    assert len(rows) == result["offered"] == 3
    assert len(seen) == 1
    overload = [row for row in rows if "Client overload" in row["error"]]
    assert len(overload) == 2
    by_index = {row["request_index"]: row for row in rows}
    first_scheduled = by_index[1]["scheduled_arrival_monotonic_s"]
    assert by_index[3]["scheduled_arrival_monotonic_s"] - first_scheduled == pytest.approx(0.02)
    assert by_index[3]["dispatch_started_monotonic_s"] - first_scheduled < 0.07


def test_failed_warmup_aborts_preserves_plan_and_checksums(tmp_path, monkeypatch, config):
    """Missing telemetry in the first warmup must not consume the full campaign."""
    import hashlib  # pylint: disable=import-outside-toplevel

    def failed_request(spec, model, deadline, timeout_s, trace_path, scheduled, dispatch):
        return workload.failed_record(spec, scheduled, "Engine timestamp contract: missing trace", dispatch)

    monkeypatch.setattr(workload, "execute_request", failed_request)
    output = tmp_path / "results"
    assert workload.run_workload(output, "model", 2026111402, config, tmp_path / "missing") == 1
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["offer_status"] == "OFFERED"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["experiment_valid"] is False
    assert "Warmup contract failed" in summary["abort_reason"]
    assert summary["offered"] == 1
    plan = json.loads((output / "plan.json").read_text())
    planned = [row for series in plan["series"] for row in series["requests"]]
    assert len(planned) == summary["expected"]
    assert sum(row["offer_status"] == "NOT_OFFERED" for row in planned) == summary["not_offered"]
    checksums = json.loads((output / "manifest.json").read_text())["sha256"]
    assert all(hashlib.sha256((output / name).read_bytes()).hexdigest() == digest for name, digest in checksums.items())
