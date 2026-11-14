"""Local-only frontend measurement contracts; no GPU/provider operations."""

import copy

# pytest injects fixtures by parameter name.
# pylint: disable=redefined-outer-name
import io
import json
import random
import threading
import time
from collections import Counter, defaultdict

import pytest

from tools import runpod_frontend_workload as workload


@pytest.fixture
def config():
    """A small paired profile with two independently seeded arrival blocks."""
    return {
        "schema_version": 1,
        "input_lengths": [128, 1024],
        "probe_output_tokens": 32,
        "idle_pairs_per_block": 2,
        "loaded_pairs_per_block": 1,
        "background_input_tokens": 1024,
        "background_output_tokens": 512,
        "probe_offsets_s": [0.01, 0.02],
        "open_loop_rates": [0.5, 0.8],
        "open_loop_seconds": 20,
        "blocks": [{"block_id": "a", "arrival_seed": 75101}, {"block_id": "b", "arrival_seed": 75102}],
        "warmup_cells": [[128, 32], [1024, 32], [4096, 32]],
        "workload_timeout_seconds": 2400,
        "request_timeout_seconds": 120,
        "max_in_flight": 64,
        "max_requests": 1000,
    }


def test_plan_counts_pairs_prompts_and_order(config):
    """Every pair compares identical tokens and retains explicit conditions."""
    plan = workload.generate_plan(config, 123)
    assert plan == workload.generate_plan(config, 123)
    assert len(plan) == 3 + 2 * (2 * 2 * 2 + 2 * 1 + 2)
    rows = [row for series in plan for row in series["requests"]]
    assert [row["request_index"] for row in rows] == list(range(1, len(rows) + 1))
    assert [row["request_id"] for row in rows] == [f"mqfront-{i}" for i in range(1, len(rows) + 1)]
    assert Counter(row["condition"] for row in rows)["idle"] == 16
    assert Counter(row["condition"] for row in rows)["loaded"] == 12
    pairs = defaultdict(list)
    prompt_hashes = defaultdict(set)
    for row in rows:
        assert row["seed"] == row["prompt_seed"] == 123 + row["input_tokens"]
        prompt_hashes[row["input_tokens"]].add(workload.prompt_ids(row["seed"], row["input_tokens"])[1])
        assert row["split"] in ("warmup", "diagnostic")
        assert row["concurrency"] == {"isolated": 1, "burst": 3, "open_loop": None}[row["mode"]]
        if row["role"] == "probe":
            pairs[row["pair_id"]].append(row)
    assert all(len(hashes) == 1 for hashes in prompt_hashes.values())
    assert all(len(pair) == 2 and {r["trace_enabled"] for r in pair} == {False, True} for pair in pairs.values())
    assert {pair[0]["trace_enabled"] for pair in pairs.values()} == {False, True}
    for series in plan:
        assert set(series) == {"series_id", "duration_s", "block_id", "condition", "requests"} | (
            {"pair_id"} if series["condition"] in ("idle", "loaded") else set()
        )
        if series["condition"] == "idle":
            assert len(series["requests"]) == 1
        if series["condition"] == "loaded":
            background, *probes = series["requests"]
            assert background["role"] == "background" and background["trace_enabled"] is False
            assert [r["arrival_offset_s"] for r in probes] == config["probe_offsets_s"]
    open_series = [series for series in plan if series["condition"] == "open_loop"]
    assert [series["requests"][0]["request_rate"] if series["requests"] else None for series in open_series] == [
        0.5,
        0.8,
        0.8,
        0.5,
    ]


def test_arrivals_exact_independent_exponential(config):
    """Arrival generation uses original rate index even in reversed blocks."""
    config["open_loop_seconds"] = 20
    plan = workload.generate_plan(config, 123)
    for series in (s for s in plan if s["condition"] == "open_loop"):
        block = next(b for b in config["blocks"] if b["block_id"] == series["block_id"])
        rate = series["requests"][0]["request_rate"]
        generator = random.Random(123 + block["arrival_seed"] + config["open_loop_rates"].index(rate) * 1000)
        expected = []
        offset = generator.expovariate(rate)
        while offset < 20:
            expected.append(offset)
            offset += generator.expovariate(rate)
        assert [row["arrival_offset_s"] for row in series["requests"]] == expected
        assert series["duration_s"] == 20


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("max_requests", True),
        ("max_in_flight", 65),
        ("max_requests", 1001),
        ("input_lengths", [True]),
        ("input_lengths", [128, 128]),
        ("open_loop_rates", [float("nan")]),
        ("open_loop_rates", [-1]),
        ("open_loop_seconds", float("inf")),
        ("probe_offsets_s", [2, 1]),
        ("probe_offsets_s", [1]),
        ("warmup_cells", [[128, True]]),
        ("warmup_cells", []),
        ("blocks", [{"block_id": "a", "arrival_seed": 1}, {"block_id": "a", "arrival_seed": 2}]),
        ("blocks", [{"block_id": "a", "arrival_seed": 1}, {"block_id": "b", "arrival_seed": 1}]),
    ],
)
def test_bad_profiles(config, field, value):
    """Reject invalid values before network or executor work."""
    config[field] = value
    with pytest.raises(ValueError):
        workload.generate_plan(config, 123)


def test_request_cap(config):
    """The full plan, including warmups, must fit the hard request cap."""
    config["max_requests"] = 4
    with pytest.raises(ValueError, match="max_requests"):
        workload.generate_plan(config, 123)


@pytest.fixture
def trace_record():
    """A client, frontend and engine monotonic chain with early transport body."""
    return {
        "request_id": "mqfront-1",
        "trace_enabled": True,
        "success": True,
        "error": None,
        "client_started_monotonic_s": 1.0,
        "latency_to_done_s": 10,
        "server_timestamps": {"queued_ts": 7, "scheduled_ts": 8, "first_token_ts": 9},
        "events": [{"raw_data": "retained"}],
    }


@pytest.fixture
def stamps():
    """The wrapper's public schema."""
    return {
        "schema_version": 1,
        "timestamp_clock": "monotonic",
        "request_id": "mqfront-1",
        "http_enter_ts": 2,
        "body_received_ts": 3,
        "handler_enter_ts": 4,
        "render_enter_ts": 5,
        "render_exit_ts": 6,
        "first_response_body_ts": 6.5,
        "response_status": 200,
        "completed_response_body": True,
        "exception": None,
    }


def test_trace_exact_join_and_transport_body(tmp_path, trace_record, stamps):
    """Transport may start before generated content; off requests never substitute a trace."""
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(stamps) + "\n", encoding="utf-8")
    assert workload.attach_frontend_trace(trace_record, path)["success"]
    assert trace_record["frontend_trace"] == stamps
    off = {**trace_record, "trace_enabled": False}
    assert workload.attach_frontend_trace(off, path)["frontend_trace"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("timestamp_clock", "wall"),
        ("request_id", "mqfront-2"),
        ("http_enter_ts", 0),
        ("body_received_ts", float("nan")),
        ("render_exit_ts", 8),
        ("first_response_body_ts", 12),
        ("response_status", 500),
        ("completed_response_body", False),
        ("exception", "RuntimeError"),
        ("instrumentation_error", "broken"),
    ],
)
def test_trace_failures_preserve_evidence(tmp_path, trace_record, stamps, field, value):
    """Malformed/mismatched/incomplete records invalidate but retain original SSE and engine data."""
    stamps[field] = value
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(stamps) + "\n", encoding="utf-8")
    before = copy.deepcopy(trace_record)
    result = workload.attach_frontend_trace(trace_record, path)
    assert not result["success"] and "Frontend timestamp contract" in result["error"]
    assert result["events"] == before["events"] and result["server_timestamps"] == before["server_timestamps"]


def test_trace_missing_duplicate_malformed(tmp_path, trace_record, stamps):
    """Missing and ambiguous joins fail closed."""
    path = tmp_path / "trace.jsonl"
    for text in (None, "not JSON\n", json.dumps(stamps) + "\n" + json.dumps(stamps) + "\n"):
        if text is not None:
            path.write_text(text, encoding="utf-8")
        row = {**trace_record, "error": None, "success": True}
        assert not workload.attach_frontend_trace(row, path)["success"]


def test_worker_private_metadata_headers_sse(monkeypatch, capsys):
    """The worker strips private metadata, bypasses proxies and preserves legacy SSE parsing."""
    captured = []

    class Response(io.BytesIO):
        """An in-memory streaming response."""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    class Opener:
        """Inspect the loopback request without opening a socket."""

        def open(self, request, timeout):
            """Return deterministic legacy events."""
            captured.append((request, timeout))
            return Response(b'data: {"choices":[{"text":""}]}\n\ndata: {"choices":[{"text":"x"}]}\n\ndata: [DONE]\n\n')

    def build(*handlers):
        assert handlers[0].proxies == {}
        assert handlers[1].redirect_request(None, None, None, None, None, None) is None
        return Opener()

    monkeypatch.setattr(workload.urllib.request, "build_opener", build)
    for enabled in (True, False):
        payload = {"model": "m", "_mq_trace": enabled, "_mq_request_id": "mqfront-1"}
        monkeypatch.setattr(workload.sys, "stdin", io.StringIO(json.dumps(payload)))
        workload.request_worker()
        items = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert list(items[0]) == ["worker_started_monotonic_s"]
        assert items[1]["client_json_ready_ts"] >= items[0]["worker_started_monotonic_s"]
        assert "client_headers_received_ts" in items[2]
        events = [item["event"] for item in items if "event" in item]
        summary = workload.summarize_events(events)
        assert summary["client_ttft_s"] == events[1]["elapsed_s"] and summary["done"]
        request, timeout = captured[-1]
        assert request.full_url == "http://127.0.0.1:8000/v1/completions" and timeout == 120
        assert json.loads(request.data) == {"model": "m"}
        assert request.get_header("X-mq-trace-id") == ("mqfront-1" if enabled else None)


def test_series_drains_full_window_and_reports_not_offered(monkeypatch, config):
    """Pool saturation remains NOT_OFFERED, with no fabricated request failure."""
    config["max_in_flight"] = 1
    rows = workload.generate_plan(config, 123)[0]["requests"]
    second = {**rows[0], "request_index": 2, "request_id": "mqfront-2"}
    series = {"series_id": "test", "duration_s": 0.08, "requests": rows + [second]}
    records = []
    gate = threading.Event()

    def execute(spec, *_args):
        gate.wait(0.03)
        return {**spec, "success": True}

    monkeypatch.setattr(workload, "execute_request", execute)
    started = time.monotonic()
    result = workload.run_series(series, "m", config, "unused", started + 1, records.append)
    assert time.monotonic() - started >= 0.08
    assert len(records) == 1 and result["failed"] == 0
    assert result["not_offered"][0]["offer_status"] == "NOT_OFFERED"


def test_actual_profile_hand_counts():
    """Freeze the authorized measurement profile's independently counted design size."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    profile = json.loads(Path("configs/runpod/frontend-v1-workload.json").read_text(encoding="utf-8"))
    plan = workload.generate_plan(profile, 2026111404)
    rows = [r for series in plan for r in series["requests"]]
    assert len(plan) == 108 and len(rows) == 613
    assert Counter(r["condition"] for r in rows) == {"warmup": 4, "idle": 80, "loaded": 60, "open_loop": 469}
    assert Counter(r["trace_enabled"] for r in rows) == {True: 533, False: 80}
    assert [
        (s["block_id"], s["requests"][0]["request_rate"], len(s["requests"]))
        for s in plan
        if s["condition"] == "open_loop"
    ] == [("a", 0.5, 91), ("a", 0.8, 143), ("b", 0.8, 148), ("b", 0.5, 87)]


def test_execute_retains_events_and_engine_on_frontend_failure(monkeypatch, tmp_path, config, stamps):
    """A frontend contract error must never erase engine events or legacy SSE."""
    spec = workload.generate_plan(config, 123)[0]["requests"][0]
    chunk = {
        "id": "cmpl-mqfront-1",
        "choices": [{"text": "content"}],
        "usage": {"prompt_tokens": 128, "completion_tokens": 32},
        "metrics": {"queue_time_ms": 1, "time_to_first_token_ms": 1, "generation_time_ms": 1},
    }
    output = "\n".join(
        json.dumps(item)
        for item in [
            {"worker_started_monotonic_s": 1},
            {"client_json_ready_ts": 1.1},
            {"client_headers_received_ts": 6.5},
            {"event": workload.parse_sse_event(json.dumps(chunk), 8)},
            {"event": workload.parse_sse_event("[DONE]", 10)},
        ]
    )
    spawned = []

    class Process:
        """An in-memory killable child stand-in."""

        returncode = 0

        def __init__(self, args, **kwargs):
            spawned.append((args, kwargs))

        def communicate(self, payload=None, timeout=None):
            """Keep private metadata only on the local child transport."""
            assert timeout > 0
            sent = json.loads(payload)
            assert sent["_mq_trace"] is True and sent["_mq_request_id"] == "mqfront-1"
            assert sent["max_tokens"] == 32 and len(sent["prompt"]) == 128
            return output, ""

        def poll(self):
            """The fake child already exited."""
            return 0

    engine = {"queued_ts": 7, "scheduled_ts": 8, "first_token_ts": 9}

    def join(record, _path):
        record["server_timestamps"] = engine

    monkeypatch.setattr(workload.subprocess, "Popen", Process)
    monkeypatch.setattr(workload, "join_server_trace", join)
    stamps["instrumentation_error"] = ["broken"]
    path = tmp_path / "server_timestamps.jsonl"
    path.with_name("frontend_timestamps.jsonl").write_text(json.dumps(stamps) + "\n", encoding="utf-8")
    started = time.monotonic()
    record = workload.execute_request(spec, "m", started + 3, 3, path, started, started)
    assert time.monotonic() - started < 1  # malformed evidence is never retried for two seconds
    assert not record["success"] and "Frontend timestamp contract" in record["error"]
    assert len(record["events"]) == 2 and record["server_timestamps"] == engine
    assert record["client_json_ready_ts"] == 1.1
    assert spawned[0][0][-1] == "--request-worker"
    assert spawned[0][0][1].endswith("runpod_frontend_workload.py")


def test_execute_child_timeout_keeps_partial_sse_and_skips_frontend(monkeypatch, tmp_path, config):
    """An incomplete SSE child is killed; no frontend joining occurs before DONE."""
    spec = workload.generate_plan(config, 123)[0]["requests"][0]
    killed = []
    output = (
        json.dumps({"worker_started_monotonic_s": 1})
        + "\n"
        + json.dumps({"event": workload.parse_sse_event('{"choices":[{"text":"partial"}]}', 0.1)})
    )

    class Process:
        """Simulate a child which emits once and then hangs."""

        returncode = None

        def __init__(self, *_args, **_kwargs):
            pass

        def communicate(self, payload=None, timeout=None):
            """The initial communicate reaches its hard deadline."""
            if payload is not None:
                assert timeout > 0
                raise workload.subprocess.TimeoutExpired("worker", timeout)
            return output, ""

        def kill(self):
            """Prove that timeout triggers child termination."""
            killed.append(True)
            self.returncode = -9

        def poll(self):
            """Return the simulated termination status."""
            return self.returncode

    monkeypatch.setattr(workload.subprocess, "Popen", Process)

    def no_join(*_args):
        raise AssertionError("No frontend join before DONE")

    monkeypatch.setattr(workload, "attach_frontend_trace", no_join)
    started = time.monotonic()
    record = workload.execute_request(spec, "m", started + 1, 1, tmp_path / "engine", started, started)
    assert killed == [True] and len(record["events"]) == 1
    assert not record["success"] and "deadline" in record["error"]
    assert record["frontend_trace"] is None


def test_workload_warmup_abort_keeps_progress_manifest(monkeypatch, tmp_path, config):
    """A failed warmup stops measurement and retains reviewable, checksummed evidence."""
    import hashlib  # pylint: disable=import-outside-toplevel

    invoked = []

    def run(series, _model, _config, _trace, _deadline, emit):
        invoked.append(series["series_id"])
        row = workload.failed_record(series["requests"][0], 1, "warmup fixture failure", 1)
        emit(row)
        return {
            "series_id": series["series_id"],
            "planned": 1,
            "offered": 1,
            "successful": 0,
            "failed": 1,
            "not_offered": [],
        }

    monkeypatch.setattr(workload, "run_series", run)
    assert workload.run_workload(tmp_path, "m", 123, config, tmp_path / "engine.jsonl") == 1
    assert len(invoked) == 1
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed"] == 1 and progress["successful"] == 0
    assert progress["experiment_valid"] is False and "Warmup contract" in progress["abort_reason"]
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    rows = [row for series in plan["series"] for row in series["requests"]]
    assert rows[0]["offer_status"] == "OFFERED" and rows[1]["offer_status"] == "NOT_OFFERED"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["sha256"]) == {"plan.json", "requests.jsonl", "series.json", "progress.json", "summary.json"}
    assert all(
        hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest["sha256"].items()
    )
