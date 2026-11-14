"""Offline completeness audit of deterministic reservation-validation fixtures."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.runpod_calibration_workload import generate_plan

SPEC_FIELDS = (
    "series_id",
    "request_index",
    "request_id",
    "phase",
    "split",
    "mode",
    "repetition",
    "concurrency",
    "request_rate",
    "input_tokens",
    "output_tokens",
    "seed",
    "arrival_offset_s",
)


def _identity(spec):
    return tuple(spec.get(field) for field in SPEC_FIELDS)


def _check_specs(label, rows, expected, errors):
    for row in rows:
        missing_fields = [field for field in SPEC_FIELDS if field not in row]
        if missing_fields:
            errors.append(f"{label}: request spec missing fields {missing_fields}")
    actual = Counter(_identity(row) for row in rows)
    wanted = Counter(_identity(row) for row in expected)
    missing, extra = wanted - actual, actual - wanted
    if missing:
        errors.append(f"{label}: {sum(missing.values())} missing or mismatched request specs")
    if extra:
        errors.append(f"{label}: {sum(extra.values())} duplicate, unplanned or mismatched request specs")
    for field in ("request_index", "request_id"):
        values = [row.get(field) for row in rows]
        if len(set(values)) != len(values):
            errors.append(f"{label}: duplicate {field}")


def _check_fields(label, saved, expected, errors):
    for key, value in expected.items():
        if saved.get(key) != value or key not in saved:
            errors.append(f"{label}: {key} differs from raw records/plan (expected {value!r})")


def audit_fixture(directory, config, seed):
    """Return completeness, failures and file hashes without raw streaming events.

    Both saved plan and raw records must match the entire regenerated plan.
    Failed offered requests remain visible in counts and failure_details; they
    cannot produce complete_success even when the summary honestly reports them.
    Missing or malformed required files raise their normal read/JSON exceptions.
    """
    # The audit follows the evidence schema, whose top-level summary has no
    # mandatory failed/status fields; those are checked when present as well.
    # pylint: disable=too-many-locals
    directory = Path(directory)
    expected_series = generate_plan(config, seed)
    expected = [row for series in expected_series for row in series["requests"]]
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    paths = [directory / name for name in ("requests.jsonl", "requests.jsonl.gz") if (directory / name).exists()]
    errors = []
    if len(paths) != 1:
        errors.append("Expected exactly one requests.jsonl or requests.jsonl.gz")
    records = []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    saved_specs = [row for series in plan["series"] for row in series["requests"]]
    _check_specs("saved plan", saved_specs, expected, errors)
    _check_specs("raw records", records, expected, errors)
    _check_fields("plan", plan, {"seed": seed, "expected": len(expected)}, errors)
    if summary.get("config") != config or summary.get("seed") != seed:
        errors.append("summary: config or seed differs from requested plan")
    series_meta = [(s["series_id"], s["duration_s"], s["arrival_seed"]) for s in expected_series]
    saved_meta = [(s["series_id"], s.get("duration_s"), s.get("arrival_seed")) for s in plan["series"]]
    if saved_meta != series_meta:
        errors.append("saved plan: series metadata differs from regenerated plan")
    offered = sum(row.get("offer_status") == "OFFERED" for row in records)
    successful = sum(row.get("success") is True for row in records)
    measured = [row for row in records if row.get("phase") == "measured"]
    failures = []
    for row in records:
        success = row.get("success") is True
        if row.get("status") != ("succeeded" if success else "failed"):
            errors.append(f"request {row.get('request_index')}: status/success mismatch")
        if not success:
            failures.append({key: row.get(key) for key in ("request_index", "request_id", "split", "mode", "error")})
    if offered != len(expected) or any(row.get("offer_status") != "OFFERED" for row in saved_specs):
        errors.append("Not every planned request was offered")
    warmup = [row for row in records if row.get("phase") == "warmup"]
    if not warmup or any(row.get("success") is not True for row in warmup):
        errors.append("Warmup missing or unsuccessful")
    valid = len(records) == len(expected) and successful == len(expected) and offered == len(expected)
    _check_fields(
        "summary",
        summary,
        {
            "expected": len(expected),
            "completed": len(records),
            "successful": successful,
            "offered": offered,
            "not_offered": len(expected) - offered,
            "measured_completed": len(measured),
            "measured_successful": sum(r.get("success") is True for r in measured),
            "experiment_valid": valid and not summary.get("abort_reason"),
        },
        errors,
    )
    if "failed" in summary:
        _check_fields("summary", summary, {"failed": len(records) - successful}, errors)
    status = "complete_success" if valid and not errors and not summary.get("abort_reason") else "incomplete_or_failed"
    if "status" in summary:
        _check_fields("summary", summary, {"status": status}, errors)
    series_rows = summary.get("series", [])
    if Counter(s.get("series_id") for s in series_rows) != Counter(s["series_id"] for s in expected_series):
        errors.append("summary: series missing, duplicated or unplanned")
    for series in series_rows:
        members = [r for r in records if r.get("series_id") == series.get("series_id")]
        planned = [r for r in expected if r["series_id"] == series.get("series_id")]
        _check_fields(
            f"series {series.get('series_id')}",
            series,
            {
                "planned": len(planned),
                "offered": len(members),
                "successful": sum(r.get("success") is True for r in members),
                "failed": sum(r.get("success") is not True for r in members),
                "not_offered_request_indices": sorted(
                    set(r["request_index"] for r in planned) - set(r["request_index"] for r in members)
                ),
            },
            errors,
        )
    counts = {}
    for row in expected:
        group = counts.setdefault(row["split"], {}).setdefault(
            row["mode"], {"planned": 0, "completed": 0, "successful": 0, "failed": 0}
        )
        group["planned"] += 1
    for row in records:
        group = counts.setdefault(row.get("split"), {}).setdefault(
            row.get("mode"), {"planned": 0, "completed": 0, "successful": 0, "failed": 0}
        )
        group["completed"] += 1
        group["successful" if row.get("success") is True else "failed"] += 1
    return {
        "status": (
            "complete_success" if valid and not errors and not summary.get("abort_reason") else "incomplete_or_failed"
        ),
        "planned": len(expected),
        "completed": len(records),
        "offered": offered,
        "successful": successful,
        "failed": len(records) - successful,
        "counts_by_split_mode": counts,
        "errors": errors,
        "failure_details": failures,
        "sha256files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir())
            if path.is_file()
        },
    }
