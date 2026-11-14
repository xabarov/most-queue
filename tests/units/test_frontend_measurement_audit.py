"""Independent, entirely synthetic evidence checks for the frontend auditor."""

import copy
import hashlib
import json

import pytest

from tools import audit_frontend_measurements as auditor
from tools.runpod_frontend_workload import generate_plan

SEED = 91
ARTIFACTS = ("plan.json", "requests.jsonl", "series.json", "progress.json", "summary.json")


def hand_trace(request_id="mqfront-1", start=100.0):
    """Construct the measured clock chain without using auditor calculations."""
    return {
        "request_id": request_id,
        "client_started_monotonic_s": start,
        "client_json_ready_ts": start + 0.005,
        "client_headers_received_ts": start + 0.09,
        "metrics": {"queue_time_ms": 20.0, "time_to_first_token_ms": 100.0, "generation_time_ms": 400.0},
        "client_ttft_s": 0.22,
        "latency_to_done_s": 0.65,
        "frontend_trace": {
            "request_id": request_id,
            "schema_version": 1,
            "timestamp_clock": "monotonic",
            "http_enter_ts": start + 0.01,
            "body_received_ts": start + 0.02,
            "handler_enter_ts": start + 0.03,
            "render_enter_ts": start + 0.04,
            "render_exit_ts": start + 0.06,
            "first_response_body_ts": start + 0.11,
            "response_status": 200,
            "completed_response_body": True,
            "exception": None,
            "render_success": True,
        },
        "server_timestamps": {
            "request_id": "cmpl-" + request_id + "-0",
            "timestamp_clock": "monotonic",
            "arrival_time_clock": "wall",
            "arrival_time": 1777777777.0,
            "queued_ts": start + 0.08,
            "scheduled_ts": start + 0.10,
            "first_token_ts": start + 0.20,
            "last_token_ts": start + 0.60,
            "num_generation_tokens": 2,
            "num_preemptions": 0,
        },
    }


def test_independent_hand_trace_nine_interval_identity():
    """Nine independently specified differences must partition the measured TTFT."""
    values = auditor.intervals(hand_trace())
    assert values == pytest.approx(
        {
            "client_to_http": 0.01,
            "receive_body": 0.01,
            "parse_validate_route": 0.01,
            "handler_to_render": 0.01,
            "render": 0.02,
            "render_to_engine_queue": 0.02,
            "queue": 0.02,
            "scheduled_to_first": 0.10,
            "first_output_delivery": 0.02,
        }
    )
    assert sum(values.values()) == pytest.approx(0.22)


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("frontend_trace", "request_id", "mqfront-2"),
        ("frontend_trace", "schema_version", 2),
        ("frontend_trace", "schema_version", True),
        ("frontend_trace", "timestamp_clock", "wall"),
        ("frontend_trace", "response_status", 500),
        ("frontend_trace", "completed_response_body", False),
        ("frontend_trace", "exception", "ValueError"),
        ("frontend_trace", "instrumentation_error", ["contract"]),
        ("frontend_trace", "instrumentation_error", []),
        ("frontend_trace", "render_exception", "ValueError"),
        ("frontend_trace", "body_received_ts", 99.0),
        ("frontend_trace", "first_response_body_ts", 100.01),
        ("server_timestamps", "scheduled_ts", 99.0),
        ("server_timestamps", "timestamp_clock", "wall"),
        (None, "client_ttft_s", -0.1),
        (None, "latency_to_done_s", -0.1),
        (None, "client_json_ready_ts", 101.0),
        (None, "client_headers_received_ts", 101.0),
    ],
)
def test_interval_contract_rejects_corrupt_evidence(group, key, value):
    """Wrong identities, clocks, outcomes and temporal order cannot yield intervals."""
    record = hand_trace()
    (record if group is None else record[group])[key] = value
    with pytest.raises((ValueError, KeyError, TypeError)):
        auditor.intervals(record)


@pytest.mark.parametrize(
    "group,key",
    [
        (None, "client_started_monotonic_s"),
        (None, "client_ttft_s"),
        (None, "latency_to_done_s"),
        (None, "client_json_ready_ts"),
        (None, "client_headers_received_ts"),
        *[
            ("frontend_trace", key)
            for key in (
                "schema_version",
                "timestamp_clock",
                "request_id",
                "response_status",
                "completed_response_body",
                "http_enter_ts",
                "body_received_ts",
                "handler_enter_ts",
                "render_enter_ts",
                "render_exit_ts",
                "first_response_body_ts",
            )
        ],
        *[("server_timestamps", key) for key in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")],
    ],
)
def test_interval_contract_rejects_missing_fields(group, key):
    """Missing marks and response metadata must never be inferred."""
    record = hand_trace()
    del (record if group is None else record[group])[key]
    with pytest.raises((ValueError, KeyError, TypeError)):
        auditor.intervals(record)


@pytest.mark.parametrize(
    "group,key",
    [
        (None, "client_started_monotonic_s"),
        (None, "client_ttft_s"),
        (None, "latency_to_done_s"),
        (None, "client_json_ready_ts"),
        (None, "client_headers_received_ts"),
        *[
            ("frontend_trace", key)
            for key in (
                "http_enter_ts",
                "body_received_ts",
                "handler_enter_ts",
                "render_enter_ts",
                "render_exit_ts",
                "first_response_body_ts",
            )
        ],
        *[("server_timestamps", key) for key in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")],
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0, True])
def test_interval_contract_rejects_nonfinite_negative_and_bool_clocks(group, key, value):
    """All recorded clocks and durations require finite positive numeric values."""
    record = hand_trace()
    (record if group is None else record[group])[key] = value
    with pytest.raises((ValueError, KeyError, TypeError)):
        auditor.intervals(record)


def write_json(path, value):
    """Persist one ordinary synthetic artifact."""
    path.write_text(json.dumps(value), encoding="utf-8")


def rehash(directory):
    """Hash exactly the five artifacts exported by the real measurement writer."""
    write_json(
        directory / "manifest.json",
        {"sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ARTIFACTS}},
    )


def write_jsonl(path, rows):
    """Write complete synthetic records, preserving failed rows."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture(name="export")
def synthetic_export(tmp_path, monkeypatch):
    """Provide complete tiny paired and loaded series with independent raw joins."""
    config = {
        "schema_version": 1,
        "input_lengths": [4],
        "probe_output_tokens": 2,
        "idle_pairs_per_block": 1,
        "loaded_pairs_per_block": 1,
        "background_input_tokens": 4,
        "background_output_tokens": 8,
        "probe_offsets_s": [0.3, 1.3],
        "open_loop_rates": [0.01],
        "open_loop_seconds": 0.1,
        "blocks": [{"block_id": "synthetic", "arrival_seed": 11}],
        "warmup_cells": [[4, 2]],
        "workload_timeout_seconds": 100,
        "request_timeout_seconds": 10,
        "max_in_flight": 3,
        "max_requests": 100,
    }
    full = generate_plan(config, SEED)
    plan = [series for series in full if series["condition"] in ("idle", "loaded")]
    monkeypatch.setattr(auditor, "generate_plan", lambda actual_config, seed: copy.deepcopy(plan))
    directory = tmp_path / "export"
    directory.mkdir()
    rows, results = [], []
    for ordinal, series in enumerate(plan):
        start = 100.0 + 10 * ordinal
        for spec in series["requests"]:
            request_start = start + spec["arrival_offset_s"]
            row = {
                **copy.deepcopy(spec),
                **hand_trace(spec["request_id"], request_start),
                "success": True,
                "done": True,
                "offer_status": "OFFERED",
                "error": None,
                "status": "succeeded",
                "usage": {"prompt_tokens": spec["input_tokens"], "completion_tokens": spec["output_tokens"]},
                "scheduled_arrival_monotonic_s": request_start,
                "dispatch_started_monotonic_s": request_start,
                "client_elapsed_s": 0.65,
            }
            row["server_timestamps"]["num_generation_tokens"] = spec["output_tokens"]
            if not spec["trace_enabled"]:
                row["frontend_trace"] = None
            if spec["role"] == "background":
                row["server_timestamps"]["last_token_ts"] = start + 2.5
                row["metrics"]["generation_time_ms"] = 2300.0
                row["latency_to_done_s"] = 2.55
                row["client_elapsed_s"] = 2.55
            rows.append(row)
        n = len(series["requests"])
        results.append(
            {
                "series_id": series["series_id"],
                "planned": n,
                "offered": n,
                "successful": n,
                "failed": 0,
                "not_offered": [],
                "not_offered_request_indices": [],
                "deadline_exhausted": False,
                "scheduled_start_monotonic_s": start,
                "scheduled_end_monotonic_s": start + series["duration_s"],
                "completed_monotonic_s": start + 3.0,
            }
        )
    n = len(rows)
    summary = {
        "model": "synthetic-model",
        "seed": SEED,
        "config": config,
        "endpoint": "http://127.0.0.1:8000/v1/completions",
        "python": "synthetic",
        "expected": n,
        "completed": n,
        "successful": n,
        "offered": n,
        "not_offered": 0,
        "experiment_valid": True,
        "deadline_exhausted": False,
        "abort_reason": None,
        "measured_completed": n,
        "measured_successful": n,
        "elapsed_s": 23.0,
        "series": results,
    }
    write_json(directory / "plan.json", {"seed": SEED, "expected": n, "series": plan})
    write_jsonl(directory / "requests.jsonl", rows)
    write_json(directory / "series.json", results)
    write_json(directory / "summary.json", summary)
    write_json(directory / "progress.json", summary)
    write_jsonl(tmp_path / "server_timestamps.jsonl", [row["server_timestamps"] for row in rows])
    write_jsonl(tmp_path / "frontend_timestamps.jsonl", [row["frontend_trace"] for row in rows if row["trace_enabled"]])
    rehash(directory)
    return directory, config, rows


def test_complete_tiny_export_passes(export):
    """Successful synthetic evidence remains explicitly diagnostic."""
    directory, config, rows = export
    result = auditor.audit(directory, config, SEED)
    assert result["status"] == "complete_success", result
    assert result["planned"] == result["completed"] == len(rows)
    assert result["instrumented_validated"] == sum(row["trace_enabled"] for row in rows)
    assert result["errors"] == result["condition_failures"] == []
    assert "no model-v3 validation" in result["data_role"]


@pytest.mark.parametrize("name", [*ARTIFACTS, "manifest.json"])
def test_missing_export_files_fail_closed(export, name):
    """An export with any missing artifact must not be declared complete."""
    directory, config, _rows = export
    (directory / name).unlink()
    assert auditor.audit(directory, config, SEED)["status"] != "complete_success"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "not_offered", "altered_spec"])
def test_raw_requests_retained_and_invalidated(export, mutation):
    """Missing, repeated, failed and changed requests cannot disappear from auditing."""
    directory, config, rows = export
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "failed":
        rows[0].update(success=False, error="synthetic failure", done=False)
    elif mutation == "not_offered":
        rows[0]["offer_status"] = "NOT_OFFERED"
    else:
        rows[0]["input_tokens"] += 1
    write_jsonl(directory / "requests.jsonl", rows)
    rehash(directory)
    result = auditor.audit(directory, config, SEED)
    assert result["status"] != "complete_success"
    assert result["completed"] == len(rows)
    assert result["errors"]


def test_altered_artifact_hash_fails(export):
    """Changing bytes without updating the manifest invalidates the export."""
    directory, config, _rows = export
    path = directory / "requests.jsonl"
    path.write_text(path.read_text().replace("\n", " \n"))
    result = auditor.audit(directory, config, SEED)
    assert result["status"] != "complete_success"
    assert any("hash mismatch" in error for error in result["errors"])


@pytest.mark.parametrize(
    "raw_name,joined_name",
    [
        ("server_timestamps.jsonl", "server_timestamps"),
        ("frontend_timestamps.jsonl", "frontend_trace"),
    ],
)
def test_raw_trace_join_mismatch_fails(export, raw_name, joined_name):
    """Matching IDs cannot hide altered raw timestamps in a joined record."""
    directory, config, rows = export
    selected = next(row for row in rows if row["trace_enabled"])
    selected[joined_name]["queued_ts" if joined_name == "server_timestamps" else "body_received_ts"] += 0.001
    write_jsonl(directory / "requests.jsonl", rows)
    rehash(directory)
    result = auditor.audit(directory, config, SEED)
    assert result["status"] != "complete_success"
    assert any("joined" in error and "differs" in error for error in result["errors"])
    assert (directory.parent / raw_name).is_file()


def test_loaded_condition_failure_keeps_successful_rows(export):
    """A failed background-overlap condition must retain all successful evidence."""
    directory, config, rows = export
    background = next(row for row in rows if row["role"] == "background")
    background["server_timestamps"]["last_token_ts"] = background["client_started_monotonic_s"] + 0.25
    background["metrics"]["generation_time_ms"] = 50.0
    write_jsonl(directory / "requests.jsonl", rows)
    write_jsonl(directory.parent / "server_timestamps.jsonl", [row["server_timestamps"] for row in rows])
    rehash(directory)
    result = auditor.audit(directory, config, SEED)
    assert result["status"] != "complete_success"
    assert result["successful"] == result["completed"] == len(rows)
    assert any("background_not_decoding" in failure for failure in result["condition_failures"])


@pytest.mark.parametrize("mutation", ["missing_key", "extra_key"])
def test_manifest_requires_exact_five_artifact_keys(export, mutation):
    """A digest map cannot omit evidence or smuggle in another artifact."""
    directory, config, _rows = export
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "missing_key":
        manifest["sha256"].pop("series.json")
    else:
        manifest["sha256"]["unexpected.json"] = "0" * 64
    write_json(path, manifest)
    assert auditor.audit(directory, config, SEED)["status"] != "complete_success"


@pytest.mark.parametrize("raw_name", ["server_timestamps.jsonl", "frontend_timestamps.jsonl"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_raw_trace_identity_completeness(export, raw_name, mutation):
    """Every required raw trace must have exactly one row for each planned ID."""
    directory, config, _rows = export
    path = directory.parent / raw_name
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    else:
        rows.append(copy.deepcopy(rows[0]))
    write_jsonl(path, rows)
    result = auditor.audit(directory, config, SEED)
    assert result["status"] != "complete_success"
    assert any("identity/count mismatch" in error for error in result["errors"])


@pytest.mark.parametrize("mutation", ["request_spec", "series_spec", "seed"])
def test_saved_plan_is_recomputed_independently(export, mutation):
    """Rehashing an altered plan cannot make it match the independently fixed plan."""
    directory, config, _rows = export
    path = directory / "plan.json"
    plan = json.loads(path.read_text())
    if mutation == "request_spec":
        plan["series"][0]["requests"][0]["output_tokens"] += 1
    elif mutation == "series_spec":
        plan["series"][0]["duration_s"] += 1
    else:
        plan["seed"] += 1
    write_json(path, plan)
    rehash(directory)
    result = auditor.audit(directory, config, SEED)
    assert result["status"] != "complete_success"
    assert any("plan" in error and "differs" in error for error in result["errors"])


@pytest.mark.parametrize("mutation", ["seed", "config", "successful", "experiment_valid"])
def test_summary_cannot_claim_another_experiment_or_hide_failure(export, mutation):
    """Valid hashes cannot repair contradictory summary provenance or counts."""
    directory, config, _rows = export
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    if mutation == "config":
        summary["config"]["probe_output_tokens"] += 1
    elif mutation == "experiment_valid":
        summary[mutation] = False
    else:
        summary[mutation] += 1
    write_json(path, summary)
    rehash(directory)
    assert auditor.audit(directory, config, SEED)["status"] != "complete_success"
