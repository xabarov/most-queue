"""Focused parser, fixed-profile and deadline tests without HTTP or GPU access."""

import hashlib
import json
import subprocess
import threading
import time

import pytest

from tools import runpod_workload as workload


def event(chunk, elapsed):
    """Build a parsed SSE event from a public synthetic chunk."""
    return workload.parse_sse_event(json.dumps(chunk), elapsed)


def final_events():
    """Provide usage, metrics and DONE independent of text chunk boundaries."""
    return [
        event({"id": "cmpl-test", "choices": []}, 0.1),
        event({"choices": [{"text": "several tokens together"}]}, 0.2),
        event({"choices": [{"text": "more text"}]}, 0.3),
        event(
            {
                "choices": [],
                "usage": {"prompt_tokens": 128, "completion_tokens": 17, "total_tokens": 145},
                "metrics": {
                    "queue_time_ms": 2.0,
                    "time_to_first_token_ms": 7.0,
                    "generation_time_ms": 100.0,
                    "mean_itl_ms": None,
                },
            },
            0.4,
        ),
        workload.parse_sse_event("[DONE]", 0.5),
    ]


def test_stream_counts_usage_not_chunks_and_preserves_server_metrics():
    """Usage-only and empty-choice events must not affect TTFT or token counts."""
    result = workload.summarize_events(final_events())
    assert result["success"]
    assert result["completion_tokens"] == 17
    assert result["client_ttft_s"] == 0.2
    assert result["latency_to_done_s"] == 0.5
    assert result["metrics"] == {
        "queue_time_ms": 2.0,
        "time_to_first_token_ms": 7.0,
        "generation_time_ms": 100.0,
        "mean_itl_ms": None,
    }
    assert result["server_request_id"] == "cmpl-test"


@pytest.mark.parametrize("chunks", [[], [event({}, 0.1)], [workload.parse_sse_event("[DONE]", 0.2)]])
def test_missing_fields_are_not_success(chunks):
    """Missing data remains missing instead of being inferred from content."""
    result = workload.summarize_events(chunks)
    assert not result["success"]
    assert result["completion_tokens"] is None
    assert result["client_ttft_s"] is None


@pytest.mark.parametrize("key", ["usage", "metrics"])
def test_missing_final_report_rejects_success(key):
    """DONE alone cannot make a request complete."""
    events = final_events()
    del events[-2]["chunk"][key]
    assert not workload.summarize_events(events)["success"]


def test_error_and_malformed_events_retained():
    """Errors remain visible even alongside otherwise complete reports."""
    error_event = event({"error": {"message": "synthetic failure"}}, 0.01)
    malformed = workload.parse_sse_event("{bad json", 0.02)
    assert malformed["raw_data"] == "{bad json"
    assert malformed["parse_error"]
    result = workload.summarize_events([error_event, malformed] + final_events())
    assert not result["success"]
    assert result["error"] == {"message": "synthetic failure"}


def test_empty_content_is_not_ttft():
    """Empty text, missing choices and usage-only chunks are valid stream events."""
    events = [event({"choices": [{"text": ""}]}, 0.01), event({"usage": None}, 0.02)] + final_events()
    assert workload.summarize_events(events)["client_ttft_s"] == 0.2


def test_exact_profile_and_reproducible_prompts():
    """The workload contains 3 warmups and 78 measured requests in fixed batches."""
    batches = list(workload.pilot_batches(42))
    assert len(batches) == 21
    records = [record for batch in batches for record in batch]
    assert sum(record["phase"] == "warmup" for record in records) == 3
    assert sum(record["phase"] == "measured" for record in records) == 78
    for length in workload.INPUT_LENGTHS:
        tokens, digest = workload.prompt_ids(42, length)
        assert len(tokens) == length
        assert min(tokens) >= 1000 and max(tokens) <= 20000
        assert workload.prompt_ids(42, length) == (tokens, digest)
        assert digest == hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def test_overall_timeout_kills_reader_and_retains_partial_events(monkeypatch):
    """A reader with a live trickling socket cannot bypass the parent deadline."""

    class FakeProcess:
        """Emulate a child that emitted data but never finishes."""

        def __init__(self):
            self.returncode = None
            self.killed = False
            self.timeout = None
            self.payload = None

        def communicate(self, payload=None, timeout=None):
            """Simulate timeout followed by recovery of flushed output."""
            if not self.killed:
                self.payload = json.loads(payload)
                self.timeout = timeout
                raise subprocess.TimeoutExpired("worker", timeout)
            return json.dumps({"event": final_events()[1]}) + "\n", ""

        def kill(self):
            """Record termination."""
            self.killed = True
            self.returncode = -9

        def poll(self):
            """Report child status."""
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(workload.subprocess, "Popen", lambda *args, **kwargs: process)
    spec = list(workload.pilot_batches(42))[0][0]
    record = workload.execute_request(spec, "synthetic-model", threading.Barrier(1), time.monotonic() + 0.1)
    assert process.killed
    assert 0 < process.timeout <= 0.1
    assert record["error"] == "Overall request deadline exceeded"
    assert record["events"] == [final_events()[1]]
    assert not record["success"]
    assert process.payload["prompt"] == workload.prompt_ids(spec["seed"], 128)[0]
    assert process.payload["max_tokens"] == 128
    assert process.payload["ignore_eos"] is True
    assert process.payload["stream_options"] == {"include_usage": True}


def test_expired_deadline_does_not_start_child(monkeypatch):
    """Workload exhaustion yields retained failed records without new HTTP requests."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Child must not be started")

    monkeypatch.setattr(workload.subprocess, "Popen", forbidden)
    spec = list(workload.pilot_batches(42))[0][0]
    record = workload.execute_request(spec, "model", threading.Barrier(1), time.monotonic() - 1)
    assert not record["success"]
    assert "deadline" in record["error"]


def test_incremental_artifacts_and_manifest(tmp_path, monkeypatch):
    """Every completion updates progress and final manifest verifies final bytes."""
    completed_before_call = []

    def fake_execute(spec, _model, _barrier, _deadline):
        # Single-thread fixture batches make progress assertions deterministic.
        progress_path = tmp_path / "progress.json"
        if progress_path.exists():
            completed_before_call.append(json.loads(progress_path.read_text())["completed"])
        return {**spec, **workload.summarize_events(final_events()), "events": final_events()}

    batches = [[{"phase": "measured", "request_index": index}] for index in range(78)]
    monkeypatch.setattr(workload, "pilot_batches", lambda seed: iter(batches))
    monkeypatch.setattr(workload, "execute_request", fake_execute)
    assert workload.run_pilot(tmp_path, "model", 42) == 0
    assert completed_before_call == list(range(1, 78))
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 78
    assert json.loads((tmp_path / "progress.json").read_text())["measured_successful"] == 78
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        workload.run_pilot(tmp_path, "model", 42)


def test_complete_reports_without_done_are_unsuccessful():
    """A socket close before DONE is an incomplete stream."""
    result = workload.summarize_events(final_events()[:-1])
    assert not result["success"]
    assert not result["done"]
    assert result["latency_to_done_s"] is None


def test_worker_reads_usage_only_and_multiline_sse_without_network(monkeypatch, capsys):
    """Exercise urllib stream consumption through an in-memory response."""
    import io  # pylint: disable=import-outside-toplevel

    class FakeOpener:
        """Return public synthetic SSE bytes without opening any socket."""

        def open(self, request, timeout):
            """Check the fixed endpoint and return a context-managed byte stream."""
            assert request.full_url == workload.ENDPOINT
            assert timeout <= 120
            return io.BytesIO(
                b': comment\n\ndata: {"choices": []}\n\n'
                b'data: {"choices": [{"text": "two tokens"}]}\n\n'
                b'data: {"usage": {"completion_tokens": 2},\n'
                b'data: "metrics": {"time_to_first_token_ms": 1}}\n\n'
                b"data: [DONE]\n\n"
            )

    monkeypatch.setattr(workload.sys, "stdin", io.StringIO('{"model": "test", "prompt": [1000]}'))
    monkeypatch.setattr(workload.urllib.request, "build_opener", lambda *args: FakeOpener())
    workload.request_worker()
    emitted = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    events = [row["event"] for row in emitted if "event" in row]
    assert len(events) == 4
    assert events[-1]["done"]
    assert workload.summarize_events(events)["completion_tokens"] == 2


@pytest.mark.parametrize("field", ["queue_time_ms", "time_to_first_token_ms", "generation_time_ms"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True, "1", "missing"])
def test_required_server_timings_are_finite_nonnegative_numbers(field, value):
    """Metrics presence alone cannot establish usable timing evidence."""
    events = final_events()
    metrics = events[-2]["chunk"]["metrics"]
    if value == "missing":
        del metrics[field]
    else:
        metrics[field] = value
    result = workload.summarize_events(events)
    assert not result["success"]
    assert result["metrics"] == metrics
    assert result["client_ttft_s"] == 0.2


@pytest.mark.parametrize("count", [None, 0, -1, True, 17.0])
def test_completion_count_must_be_positive_actual_integer(count):
    """Booleans and guessed or absent counts cannot satisfy reported token usage."""
    events = final_events()
    events[-2]["chunk"]["usage"]["completion_tokens"] = count
    assert not workload.summarize_events(events)["success"]


def test_ttft_content_event_required_for_success():
    """Server scheduled-to-first timing cannot substitute for client arrival TTFT."""
    events = final_events()
    events = [events[0], *events[-2:]]
    result = workload.summarize_events(events)
    assert not result["success"]
    assert result["client_ttft_s"] is None
    assert result["metrics"] == events[-2]["chunk"]["metrics"]


@pytest.mark.parametrize(
    "prompt_count,completion_count,expected",
    [(128, 128, True), (127, 128, False), (128, 127, False), (128.0, 128, False)],
)
def test_fixed_profile_requires_exact_usage_lengths(monkeypatch, prompt_count, completion_count, expected):
    """Any server-reported length drift fails the request and retains raw evidence."""
    events = final_events()
    events[-2]["chunk"]["usage"].update(prompt_tokens=prompt_count, completion_tokens=completion_count)
    output = "\n".join(json.dumps({"event": value}) for value in events)

    class FakeProcess:
        """Return a complete synthetic stream without contacting the server."""

        returncode = 0

        def communicate(self, _payload=None, timeout=None):  # pylint: disable=unused-argument
            """Return synthetic worker output."""
            return output, ""

        def poll(self):
            """Report already completed execution."""
            return self.returncode

    monkeypatch.setattr(workload.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    spec = list(workload.pilot_batches(42))[0][0]
    result = workload.execute_request(spec, "model", threading.Barrier(1), time.monotonic() + 10)
    assert result["success"] is expected
    assert result["usage"]["prompt_tokens"] == prompt_count
    assert result["usage"]["completion_tokens"] == completion_count
    assert result["events"] == events
    if not expected:
        assert "token lengths" in result["error"]
