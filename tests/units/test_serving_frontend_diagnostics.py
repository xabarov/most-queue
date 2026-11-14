"""Public-fixture and synthetic checks for descriptive frontend diagnostics."""

import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from examples import serving_frontend_diagnostics as diagnostics
from most_queue.theory.continuous_batching.advance_reservation_calibration import AdvanceReservationFit

ROOT = Path(__file__).resolve().parents[2]
V2_FIXTURE = ROOT / "tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_reservation_v2_20261114"


def synthetic_fixture(directory):
    """Write a tiny public fixture whose whitespace must survive extraction."""
    directory.mkdir()
    raw = {
        "requests": b'{ "synthetic": "request" }\n\n',
        "server_timestamps": b'{"synthetic":"engine"}\n',
        "frontend_timestamps": b' {"synthetic":"frontend"} \n',
    }
    files = {
        name: b'{ "synthetic": true }\n'
        for name in ("plan.json", "series.json", "summary.json", "progress.json", "producer-manifest.json")
    }
    files.update({name + ".jsonl.gz": gzip.compress(data, mtime=0) for name, data in raw.items()})
    for name, data in files.items():
        (directory / name).write_bytes(data)
    manifest = {"sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return files, raw


def test_unpack_preserves_exact_raw_bytes_and_producer_layout(tmp_path):
    """Extraction restores measurements and their parent traces byte for byte."""
    fixture, target = tmp_path / "fixture", tmp_path / "restored"
    files, raw = synthetic_fixture(fixture)
    target.mkdir()
    measurements = diagnostics.unpack_fixture(fixture, target)
    assert measurements == target / "measurements"
    for name in ("plan.json", "series.json", "summary.json", "progress.json"):
        assert (measurements / name).read_bytes() == files[name]
    assert (measurements / "manifest.json").read_bytes() == files["producer-manifest.json"]
    assert (measurements / "requests.jsonl").read_bytes() == raw["requests"]
    for name in ("server_timestamps", "frontend_timestamps"):
        assert (target / (name + ".jsonl")).read_bytes() == raw[name]
        assert not (measurements / (name + ".jsonl")).exists()
    assert {name: (fixture / name).read_bytes() for name in files} == files


@pytest.mark.parametrize("attack", ["modified_hash", "modified_file", "path_traversal", "symlink"])
def test_unpack_rejects_invalid_manifest_identity(tmp_path, attack):
    """Hash, parent traversal and symlink changes fail before extraction."""
    fixture, target = tmp_path / "fixture", tmp_path / "restored"
    synthetic_fixture(fixture)
    target.mkdir()
    manifest_path = fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if attack == "modified_hash":
        manifest["sha256"]["plan.json"] = "0" * 64
    elif attack == "modified_file":
        (fixture / "requests.jsonl.gz").write_bytes(gzip.compress(b"{}\n"))
    elif attack == "path_traversal":
        outside = tmp_path / "outside.json"
        outside.write_bytes(b"{}\n")
        manifest["sha256"]["../outside.json"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    else:
        outside = tmp_path / "outside.json"
        outside.write_bytes((fixture / "plan.json").read_bytes())
        (fixture / "plan.json").unlink()
        (fixture / "plan.json").symlink_to(outside)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fixture file identity differs"):
        diagnostics.unpack_fixture(fixture, target)
    assert not (target / "measurements").exists()


def test_summarize_groups_retains_every_factor_and_observation():
    """All six grouping factors remain separate; off intervals remain absent."""
    fields = ("block_id", "condition", "role", "input_tokens", "output_tokens", "trace_enabled")
    rows = []
    combinations = list(
        itertools.product(
            ("a", "b"), ("idle", "loaded"), ("probe", "background"), (128, 1024), (32, 128), (False, True)
        )
    )
    for combination in combinations:
        for value in (1.0, 3.0):
            row = dict(zip(fields, combination))
            row.update(
                metrics={"client_ttft": value, "queue": 2 * value},
                intervals={"pre_engine": value / 2} if row["trace_enabled"] else None,
            )
            rows.append(row)
    unchanged = copy.deepcopy(rows)
    groups = diagnostics.summarize_groups(rows)
    assert rows == unchanged
    assert len(groups) == len(combinations) == 64
    assert {tuple(group[key] for key in fields) for group in groups} == set(combinations)
    assert sum(group["requests"] for group in groups) == len(rows) == 128
    for group in groups:
        assert group["requests"] == 2
        assert group["metrics"]["client_ttft"]["mean"] == 2.0
        assert group["metrics"]["queue"]["mean"] == 4.0
        assert group["metrics"]["client_ttft"]["count"] == 2
        if group["trace_enabled"]:
            assert group["intervals"]["pre_engine"]["mean"] == 1.0
        else:
            assert "intervals" not in group


def public_replay_inputs():
    """Copy only two records from one already-public v2 open-loop series."""
    records = [
        json.loads(line) for line in gzip.decompress((V2_FIXTURE / "requests.jsonl.gz").read_bytes()).splitlines()
    ]
    series_id = next(row["series_id"] for row in records if row["mode"] == "open_loop")
    selected = copy.deepcopy([row for row in records if row["series_id"] == series_id][:2])
    for row in selected:
        row.update(block_id="a", split="diagnostic")
    frozen = json.loads((V2_FIXTURE / "frozen-fit.json").read_text())
    return selected, AdvanceReservationFit(**frozen["advance_reservation"])


def update_observed_durations(row):
    """Keep imported durations consistent after a synthetic outcome change."""
    stamps = row["server_timestamps"]
    row["metrics"]["time_to_first_token_ms"] = (stamps["first_token_ts"] - stamps["scheduled_ts"]) * 1000
    row["metrics"]["generation_time_ms"] = (stamps["last_token_ts"] - stamps["first_token_ts"]) * 1000
    row["metrics"]["mean_itl_ms"] = row["metrics"]["generation_time_ms"] / (row["output_tokens"] - 1)
    row["latency_to_done_s"] = stamps["last_token_ts"] - row["client_started_monotonic_s"] + 0.25
    row["client_elapsed_s"] = row["latency_to_done_s"] + 0.1


def test_replay_preserves_source_alignment_and_nine_point_slo_grid():
    """Temporary importer adaptation preserves diagnostic source rows and IDs."""
    records, fit = public_replay_inputs()
    unchanged = copy.deepcopy(records)
    result = diagnostics.replay_series(records, fit)
    assert records == unchanged
    assert all(row["split"] == "diagnostic" for row in records)
    assert result["source_split"] == "diagnostic"
    assert result["block_id"] == "a"
    assert result["replay"]["requests"] == 2
    index_to_id = {row["request_index"]: row["request_id"] for row in records}
    assert {index_to_id[index] for index in result["replay"]["request_indices"]} == set(index_to_id.values())
    assert {row["request_index"] for row in result["chronological_errors"]} == set(index_to_id)
    grid = result["joint_client_slo"]
    assert len(grid) == 9
    assert {(point["ttft_limit_seconds"], point["tpot_limit_seconds"]) for point in grid} == set(
        itertools.product((0.25, 0.5, 1.0), (0.030, 0.035, 0.040))
    )
    assert all(point["offered"] == 2 for point in grid)


def test_replay_predictions_do_not_use_observed_token_or_client_outcomes():
    """Changing outcomes with fixed queue arrivals/lengths changes no prediction."""
    records, fit = public_replay_inputs()
    original = diagnostics.replay_series(records, fit)
    changed = copy.deepcopy(records)
    for row in changed:
        row["server_timestamps"]["first_token_ts"] += 0.125
        row["server_timestamps"]["last_token_ts"] += 0.5
        row["client_ttft_s"] += 0.25
        update_observed_durations(row)
    untouched = copy.deepcopy(changed)
    replay = diagnostics.replay_series(changed, fit)
    assert changed == untouched
    for before, after in zip(records, changed):
        assert before["server_timestamps"]["queued_ts"] == after["server_timestamps"]["queued_ts"]
        assert before["server_timestamps"]["scheduled_ts"] == after["server_timestamps"]["scheduled_ts"]
        assert (before["input_tokens"], before["output_tokens"]) == (after["input_tokens"], after["output_tokens"])
    for metric, report in original["replay"]["metrics"].items():
        assert replay["replay"]["metrics"][metric]["predicted"] == report["predicted"]
    for metric in ("client_ttft", "client_tpot", "server_ttft", "server_tpot", "server_scheduled_to_first_token"):
        assert replay["replay"]["metrics"][metric]["observed"] != original["replay"]["metrics"][metric]["observed"]
    assert replay["chronological_errors"] != original["chronological_errors"]


def test_observed_busy_periods_are_half_open_and_only_posthoc():
    """Touching observed intervals split; a tiny overlap joins, with fixed predictions."""
    records, fit = public_replay_inputs()
    records.sort(key=lambda row: row["server_timestamps"]["queued_ts"])
    boundary = records[1]["server_timestamps"]["queued_ts"]
    records[0]["server_timestamps"]["last_token_ts"] = boundary
    update_observed_durations(records[0])
    touching = diagnostics.replay_series(records, fit)
    assert [row["observed_engine_busy_period"] for row in touching["chronological_errors"]] == [1, 2]
    assert "posthoc" in touching["busy_period_scope"]
    assert "never a predictor" in touching["busy_period_scope"]
    records[0]["server_timestamps"]["last_token_ts"] += 1e-6
    update_observed_durations(records[0])
    overlapping = diagnostics.replay_series(records, fit)
    assert [row["observed_engine_busy_period"] for row in overlapping["chronological_errors"]] == [1, 1]
    for metric, report in touching["replay"]["metrics"].items():
        assert overlapping["replay"]["metrics"][metric]["predicted"] == report["predicted"]
