"""Runpod trace imports preserve offered cohorts and distinct timing bases."""

import copy
import gzip
import json
from pathlib import Path

import pytest

from most_queue.io.runpod import read_runpod_trace
from most_queue.io.serving import ServingRequest, summarize_serving_slo

FIXTURE = Path(__file__).parents[1] / "fixtures/serving_measurements/runpod_qwen3_8b_a40_20261114/requests.jsonl.gz"


@pytest.fixture
def raw():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as stream:
        row = json.loads(next(stream))
    return row


def test_frozen_pilot_preserves_all_records_and_warmup():
    cohorts = read_runpod_trace(FIXTURE, measurement_profile="qwen3-a40-pilot")
    assert len(cohorts) == 21
    assert sum(len(cohort.requests) for cohort in cohorts) == 81
    assert sum(len(cohort.requests) for cohort in cohorts if cohort.metadata["phase"] == "warmup") == 3
    measured = read_runpod_trace(str(FIXTURE), measurement_profile="qwen3-a40-pilot", include_warmup=False)
    assert len(measured) == 18
    assert sum(len(cohort.requests) for cohort in measured) == 78
    assert all(cohort.metadata["latency_basis"] == "client_to_done" for cohort in measured)
    assert all(value is None for cohort in cohorts for value in cohort.metadata["server_arrival_times"])
    assert all(cohort.metadata["mode"] == "isolated" for cohort in measured if len(cohort.requests) == 1)
    assert all(row.server_queue_time is not None and row.server_queue_time > 0 for c in cohorts for row in c.requests)


def test_timing_units_and_usage_do_not_depend_on_chunk_count(raw):
    raw["events"] = [{"chunk": {"choices": [{"text": "many tokens in one event"}]}}]
    cohort = read_runpod_trace([raw], measurement_profile="test")[0]
    request = cohort.requests[0]
    assert request.output_tokens == 128
    queue = raw["metrics"]["queue_time_ms"] / 1000
    scheduled = raw["metrics"]["time_to_first_token_ms"] / 1000
    generation = raw["metrics"]["generation_time_ms"] / 1000
    assert request.server_queue_time == queue
    assert request.server_scheduled_to_first_token == scheduled
    assert request.server_ttft == queue + scheduled
    assert request.server_generation_time == generation
    assert request.server_tpot == generation / 127
    assert request.ttft == raw["client_ttft_s"]
    assert request.tpot == (raw["latency_to_done_s"] - raw["client_ttft_s"]) / 127
    assert cohort.duration == pytest.approx(raw["latency_to_done_s"])
    assert cohort.metadata["request_ids"] == [raw["server_request_id"]]
    assert cohort.metadata["server_ttft_basis"] == "queue_entry_to_first_token"


def test_explicit_open_loop_mixed_lengths_actual_arrivals_and_failures(raw):
    base = dict(
        raw,
        phase="measured",
        split="validation",
        mode="open_loop",
        series_id="heldout-loaded",
        output_tokens=128,
        concurrency=None,
        request_rate=2.0,
        repetition=2,
    )
    first = dict(
        base,
        request_index=12,
        client_started_monotonic_s=10.0,
        dispatch_started_monotonic_s=9.9,
        scheduled_arrival_monotonic_s=9.8,
    )
    last = copy.deepcopy(base)
    last.update(
        request_index=14,
        input_tokens=512,
        output_tokens=64,
        client_started_monotonic_s=11.0,
        dispatch_started_monotonic_s=10.9,
        scheduled_arrival_monotonic_s=10.8,
    )
    last["usage"] = {"prompt_tokens": 512, "completion_tokens": 64, "total_tokens": 576}
    last["completion_tokens"] = 64
    failed = dict(
        base,
        request_index=13,
        success=False,
        done=False,
        error="connect failed",
        client_started_monotonic_s=None,
        dispatch_started_monotonic_s=10.5,
        scheduled_arrival_monotonic_s=10.3,
        client_ttft_s=None,
        latency_to_done_s=None,
        client_elapsed_s=20.0,
        usage=None,
        metrics=None,
    )
    cohort = read_runpod_trace([last, failed, first], measurement_profile="test")[0]
    assert cohort.metadata["request_indices"] == [12, 13, 14]
    assert cohort.metadata["arrival_times"] == [0.0, 0.5, 1.0]
    assert cohort.metadata["dispatch_lateness_s"] == pytest.approx([0.1, 0.2, 0.1])
    assert cohort.metadata["requested_inputs"] == [128, 128, 512]
    assert cohort.metadata["requested_outputs"] == [128, 128, 64]
    assert cohort.metadata["max_concurrency"] is None
    assert cohort.metadata["errors"] == [None, "connect failed", None]
    assert cohort.duration == 20.5
    failure = cohort.requests[1]
    assert failure.status == "failed"
    assert failure.input_tokens == failure.output_tokens == 0
    assert failure.latency is None
    assert failure.server_queue_time is None
    assert failure.server_ttft is None
    slo = summarize_serving_slo(cohort, ttft_limit=10, tpot_limit=10)
    assert slo["offered"] == 3 and slo["failed"] == 1
    assert slo["joint_fraction_bounds"] == [2 / 3, 2 / 3]


def test_failed_partial_observations_do_not_fabricate_queue_or_final_latency(raw):
    raw.update(success=False, done=False, error="read timeout", usage={"prompt_tokens": 128, "completion_tokens": 5})
    raw["metrics"].pop("queue_time_ms")
    cohort = read_runpod_trace([raw], measurement_profile="test")[0]
    request = cohort.requests[0]
    assert request.status == "failed" and request.latency is None
    assert request.input_tokens == 128 and request.output_tokens == 5
    assert request.ttft == raw["client_ttft_s"]
    assert request.server_scheduled_to_first_token is not None
    assert request.server_queue_time is None and request.server_ttft is None
    assert request.server_tpot is None
    assert cohort.duration == pytest.approx(raw["client_elapsed_s"])


def test_failed_elapsed_window_starts_at_recorded_dispatch(raw):
    raw.update(
        success=False,
        done=False,
        error="read timeout",
        client_started_monotonic_s=10.5,
        dispatch_started_monotonic_s=10.0,
        client_elapsed_s=5.0,
    )
    cohort = read_runpod_trace([raw], measurement_profile="test")[0]
    assert cohort.duration == 4.5


@pytest.mark.parametrize(
    "change",
    [
        {"done": False},
        {"success": 1},
        {"error": "error despite success"},
        {"client_ttft_s": None},
        {"client_ttft_s": True},
        {"client_ttft_s": float("nan")},
        {"client_elapsed_s": float("inf")},
        {"latency_to_done_s": -1},
        {"latency_to_done_s": 0.01},
        {"client_started_monotonic_s": None},
        {"usage": None},
        {"usage": {"prompt_tokens": 128, "completion_tokens": 127}},
        {"usage": {"prompt_tokens": True, "completion_tokens": 128}},
        {"usage": {"prompt_tokens": 128, "completion_tokens": 128, "total_tokens": 257}},
        {"completion_tokens": 127},
        {"metrics": {}},
        {"metrics": {"queue_time_ms": 0, "time_to_first_token_ms": True, "generation_time_ms": 3}},
        {"metrics": {"queue_time_ms": 0, "time_to_first_token_ms": 1, "generation_time_ms": float("nan")}},
    ],
)
def test_invalid_successes_raise_instead_of_disappearing(raw, change):
    raw.update(change)
    with pytest.raises(ValueError):
        read_runpod_trace([raw], measurement_profile="test")


def test_duplicates_are_rejected_even_in_excluded_warmup(raw):
    with pytest.raises(ValueError, match="duplicate"):
        read_runpod_trace([raw, raw], measurement_profile="test", include_warmup=False)


@pytest.mark.parametrize("profile", [None, "", "  ", 17])
def test_explicit_profile_required(raw, profile):
    with pytest.raises(ValueError, match="measurement_profile"):
        read_runpod_trace([raw], measurement_profile=profile)


def test_series_cannot_mix_attributes(raw):
    raw.update(series_id="series", split="train", mode="burst", request_rate=None, output_tokens=128)
    other = dict(raw, request_index=2, repetition=1)
    with pytest.raises(ValueError, match="inconsistent"):
        read_runpod_trace([raw, other], measurement_profile="test")


def test_failed_request_without_any_start_is_rejected(raw):
    raw.update(success=False, done=False, client_started_monotonic_s=None, error="never launched")
    with pytest.raises(ValueError, match="timestamp"):
        read_runpod_trace([raw], measurement_profile="test")


def test_jsonl_and_corrupt_gzip_are_strict(tmp_path, raw):
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    assert len(read_runpod_trace(path, measurement_profile="test")) == 1
    path.write_text(json.dumps(raw) + "\n{corrupt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_runpod_trace(path, measurement_profile="test")
    corrupt = tmp_path / "requests.jsonl.gz"
    corrupt.write_bytes(b"invalid gzip")
    with pytest.raises(ValueError, match="cannot read"):
        read_runpod_trace(corrupt, measurement_profile="test")


def test_server_tpot_requires_completed_multiple_output_tokens():
    request = ServingRequest(1, 1, "completed", 0.1, 0.2, server_generation_time=0.1)
    assert request.server_tpot is None
    with pytest.raises(ValueError):
        ServingRequest(1, 1, "completed", 0.1, 0.2, server_generation_time=-0.1)


def _with_server_timestamps(raw, *, client_start=100.0, queued=100.3):
    row = copy.deepcopy(raw)
    row["client_started_monotonic_s"] = client_start
    values = [queued, queued + 0.1, queued + 0.3, queued + 1.5]
    row["server_timestamps"] = dict(zip(("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts"), values))
    row["server_timestamps"].update(
        request_id=row["server_request_id"] + "-0", num_generation_tokens=128, num_preemptions=0
    )
    for key, left, right in zip(("queue_time_ms", "time_to_first_token_ms", "generation_time_ms"), values, values[1:]):
        row["metrics"][key] = (right - left) * 1000
    return row


def test_server_arrivals_share_client_origin_and_preserve_client_order(raw):
    first = _with_server_timestamps(raw)
    second = _with_server_timestamps(raw, client_start=100.1, queued=100.2)
    second["request_index"] = 2
    third = dict(raw, client_started_monotonic_s=100.2, request_index=3)
    cohort = read_runpod_trace([third, second, first], measurement_profile="test")[0]
    assert cohort.metadata["request_indices"] == [1, 2, 3]
    assert cohort.metadata["arrival_times"] == pytest.approx([0, 0.1, 0.2])
    assert cohort.metadata["server_arrival_times"][:2] == pytest.approx([0.3, 0.2])
    assert cohort.metadata["server_arrival_times"][2] is None


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": "another-request-0"},
        {"queued_ts": float("nan")},
        {"queued_ts": True},
        {"queued_ts": 0},
        {"scheduled_ts": 99},
        {"first_token_ts": 100.7},
        {"last_token_ts": None},
        {"num_generation_tokens": 127},
        {"num_generation_tokens": True},
        {"num_preemptions": 1},
    ],
)
def test_completed_server_timestamps_are_strict(raw, change):
    row = _with_server_timestamps(raw)
    row["server_timestamps"].update(change)
    with pytest.raises(ValueError):
        read_runpod_trace([row], measurement_profile="test")


def test_server_timestamp_metric_difference_tolerance(raw):
    row = _with_server_timestamps(raw)
    row["metrics"]["queue_time_ms"] += 0.000009
    assert read_runpod_trace([row], measurement_profile="test")
    row["metrics"]["queue_time_ms"] += 0.000002
    with pytest.raises(ValueError, match="duration disagrees"):
        read_runpod_trace([row], measurement_profile="test")


def test_failed_offers_survive_invalid_or_partial_server_timestamps(raw):
    row = _with_server_timestamps(raw)
    row.update(success=False, done=False, error="trace incomplete")
    row["server_timestamps"] = {"request_id": "wrong-id", "queued_ts": float("nan")}
    cohort = read_runpod_trace([row], measurement_profile="test")[0]
    assert cohort.requests[0].status == "failed"
    assert cohort.metadata["server_arrival_times"] == [None]
    row["server_timestamps"] = {"request_id": row["server_request_id"] + "-0", "queued_ts": 100.3}
    cohort = read_runpod_trace([row], measurement_profile="test")[0]
    assert cohort.metadata["server_arrival_times"] == pytest.approx([0.3])
