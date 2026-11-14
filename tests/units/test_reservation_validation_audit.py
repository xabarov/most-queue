"""Completeness auditing retains every planned request, including failures."""

# Pytest injects fixture names; descriptive test names serve as test docs.
# pylint: disable=missing-function-docstring,redefined-outer-name

import copy
import gzip
import json
import shutil
from pathlib import Path

import pytest

from tools.audit_reservation_validation import SPEC_FIELDS, audit_fixture
from tools.runpod_calibration_workload import generate_plan

FIXTURE = Path(__file__).parents[1] / "fixtures/serving_measurements/runpod_qwen3_8b_a40_calibration_v1_20261114"


def configuration():
    summary = json.loads((FIXTURE / "summary.json").read_text(encoding="utf-8"))
    return summary["config"], summary["seed"]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(scope="module")
def compact_records():
    with gzip.open(FIXTURE / "requests.jsonl.gz", "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    # Streaming payloads are irrelevant to identity/count mutations. The first
    # test audits the unmodified full evidence, including its compressed file.
    for record in records:
        record.pop("events", None)
    return records


@pytest.fixture
def copied(tmp_path, compact_records):
    for name in ("summary.json", "plan.json"):
        shutil.copy(FIXTURE / name, tmp_path / name)
    return tmp_path, copy.deepcopy(compact_records)


def audit_copy(copied):
    directory, records = copied
    (directory / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return audit_fixture(directory, *configuration())


def test_preserved_v1_complete_and_hashed():
    audit = audit_fixture(FIXTURE, *configuration())
    assert audit["status"] == "complete_success"
    assert not audit["errors"]
    assert not audit["failure_details"]
    assert audit["planned"] == sum(len(s["requests"]) for s in generate_plan(*configuration()))
    assert audit["counts_by_split_mode"]["warmup"]["burst"]["successful"] > 0
    assert len(audit["sha256files"]["requests.jsonl.gz"]) == 64


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "unplanned", "plan_omit", "not_offered"])
def test_missing_duplicate_and_unplanned_requests(copied, mutation):
    directory, records = copied
    if mutation == "omit":
        records.pop()
    elif mutation == "duplicate":
        records.append(records[-1])
    elif mutation == "unplanned":
        records[-1]["request_index"] = 999999
    elif mutation == "not_offered":
        records[-1]["offer_status"] = "NOT_OFFERED"
    else:
        plan = json.loads((directory / "plan.json").read_text())
        plan["series"][-1]["requests"].pop()
        write_json(directory / "plan.json", plan)
    audit = audit_copy(copied)
    assert audit["status"] == "incomplete_or_failed"
    assert audit["errors"]


@pytest.mark.parametrize("field", SPEC_FIELDS)
def test_every_request_identity_field_is_checked(copied, field):
    copied[1][-1][field] = "mismatch"
    audit = audit_copy(copied)
    assert audit["status"] == "incomplete_or_failed"
    assert any("mismatched request specs" in error for error in audit["errors"])


@pytest.mark.parametrize("field", ["completed", "offered", "not_offered", "successful", "measured_completed"])
def test_summary_count_corruption(copied, field):
    directory, _ = copied
    summary = json.loads((directory / "summary.json").read_text())
    summary[field] += 1
    write_json(directory / "summary.json", summary)
    assert any(field in error for error in audit_copy(copied)["errors"])


def test_reported_failure_is_preserved_without_events(copied):
    directory, records = copied
    record = records[-1]
    record.update(success=False, status="failed", error="synthetic failure")
    summary = json.loads((directory / "summary.json").read_text())
    summary["successful"] -= 1
    summary["measured_successful"] -= 1
    summary["experiment_valid"] = False
    summary["series"][-1]["successful"] -= 1
    summary["series"][-1]["failed"] += 1
    write_json(directory / "summary.json", summary)
    audit = audit_copy(copied)
    assert audit["status"] == "incomplete_or_failed"
    assert audit["failed"] == 1
    assert not audit["errors"]
    assert audit["failure_details"] == [
        {key: record[key] for key in ("request_index", "request_id", "split", "mode", "error")}
    ]
    assert "events" not in json.dumps(audit)


def test_warmup_failure_is_explicit(copied):
    copied[1][0].update(success=False, status="failed", error="warmup failed")
    audit = audit_copy(copied)
    assert "Warmup missing or unsuccessful" in audit["errors"]


def test_changed_config_is_not_hardcoded_to_v1_count(copied):
    directory, _ = copied
    config, seed = configuration()
    config.update(train_repetitions=[], isolated_validation_repetitions=[1], open_loop_seeds=[64101, 64102])
    audit_copy(copied)
    audit = audit_fixture(directory, config, seed + 1)
    assert audit["planned"] == sum(len(s["requests"]) for s in generate_plan(config, seed + 1))
    assert audit["status"] == "incomplete_or_failed"


def test_fresh_synthetic_config_can_succeed_with_different_total(tmp_path):
    config, seed = configuration()
    config.update(
        input_lengths=[2],
        output_lengths=[1],
        train_repetitions=[],
        isolated_validation_repetitions=[1],
        burst_repetitions=[],
        open_loop_rates=[],
    )
    series = generate_plan(config, seed + 1)
    records = [
        dict(row, offer_status="OFFERED", success=True, status="succeeded")
        for group in series
        for row in group["requests"]
    ]
    for group in series:
        for row in group["requests"]:
            row["offer_status"] = "OFFERED"
    measured = sum(r["phase"] == "measured" for r in records)
    summary = {
        "config": config,
        "seed": seed + 1,
        "expected": len(records),
        "completed": len(records),
        "successful": len(records),
        "offered": len(records),
        "not_offered": 0,
        "measured_completed": measured,
        "measured_successful": measured,
        "experiment_valid": True,
        "failed": 0,
        "status": "complete_success",
        "series": [
            {
                "series_id": s["series_id"],
                "planned": len(s["requests"]),
                "offered": len(s["requests"]),
                "successful": len(s["requests"]),
                "failed": 0,
                "not_offered_request_indices": [],
            }
            for s in series
        ],
    }
    write_json(tmp_path / "summary.json", summary)
    write_json(tmp_path / "plan.json", {"seed": seed + 1, "expected": len(records), "series": series})
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    assert audit_fixture(tmp_path, config, seed + 1)["status"] == "complete_success"
    summary["failed"] = 1
    summary["status"] = "incorrect"
    write_json(tmp_path / "summary.json", summary)
    audit = audit_fixture(tmp_path, config, seed + 1)
    assert audit["status"] == "incomplete_or_failed"
    assert any("failed differs" in e for e in audit["errors"])
    assert any("status differs" in e for e in audit["errors"])
    del records[0]["request_rate"]  # Missing and explicit null are distinct.
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    assert any("missing fields" in e for e in audit_fixture(tmp_path, config, seed + 1)["errors"])
